"""
kd_train_student.py —— 序列级 KD 第 2 步：用教师答案对 Dense 学生做纯文本 SFT

只更新 Thinker（文本模型）与（共享的）lm_head，不计算/不更新 Talker（语音），
所以这是“纯文本蒸馏”，训练完仍是一个完整 omni 模型目录，可直接被 web demo 加载。

CPU 冒烟（几条样本、1 个 epoch，几分钟跑通）：
  python kd_train_student.py --student_dir ./minimind-3o --data kd_sft.jsonl \
      --out_dir ./kd_dense --epochs 1 --batch_size 1 --max_seq_len 256 --device cpu
GPU/云端：
  python kd_train_student.py --student_dir ./minimind-3o --data kd_sft.jsonl \
      --out_dir ./kd_dense --epochs 3 --batch_size 8 --max_seq_len 512 --device cuda --lr 1e-4
"""
import os
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from kd_common import load_model


class TextSFTJsonl(Dataset):
    """读取 kd_sft.jsonl（conversations 结构），只对 assistant 部分计算 loss。"""

    def __init__(self, jsonl_path, tokenizer, max_len=512):
        self.tok = tokenizer
        self.max_len = max_len
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line)["conversations"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        conv = self.samples[idx]
        # 完整序列（含 user+assistant）
        full_text = self.tok.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False, open_thinking=False)
        # 只到 user 的前缀（add_generation_prompt=True，模型“要开始回答”的位置）
        pref_text = self.tok.apply_chat_template(
            conv[:-1], tokenize=False, add_generation_prompt=True, open_thinking=False)
        full_ids = self.tok(full_text, add_special_tokens=False).input_ids
        pref_ids = self.tok(pref_text, add_special_tokens=False).input_ids

        # 若没有以 eos 结尾，手动补上（确保学生学会停止）
        if full_ids and full_ids[-1] != self.tok.eos_token_id:
            full_ids = full_ids + [self.tok.eos_token_id]

        full_ids = full_ids[: self.max_len]
        labels = list(full_ids)
        # 前 1..len(pref) 是 prompt，全部 mask；只对 assistant 部分（含 eos）计算 loss
        if full_ids[: len(pref_ids)] == pref_ids:
            labels[: len(pref_ids)] = [-100] * len(pref_ids)
        else:
            # 模板对不齐时的兜底：仅对末尾 assistant 文本训练（保留全部，不 crash）
            print("[data] 前缀对不齐，该样本对全部 token 计算 loss（罕见）")
        return full_ids, labels


def collate(batch, pad_id):
    maxlen = max(len(ids) for ids, _ in batch)
    xs, ys = [], []
    for ids, labels in batch:
        pad = maxlen - len(ids)
        xs.append(ids + [pad_id] * pad)
        ys.append(labels + [-100] * pad)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_dir", default="./minimind-3o", help="学生基座目录（Dense）")
    ap.add_argument("--data", default="kd_sft.jsonl", help="教师答案 SFT jsonl")
    ap.add_argument("--out_dir", default="./kd_dense", help="蒸馏后模型输出目录")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1, help="梯度累积步数（CPU 用大累积等效大 batch）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（0=全部）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, tok = load_model(args.student_dir, args.device, args.dtype)
    ds = TextSFTJsonl(args.data, tok, args.max_seq_len)
    if args.limit:
        ds.samples = ds.samples[: args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    print(f"[student] 训练样本={len(ds)} 设备={args.device} 批次={len(loader)}")

    # 只训练 Thinker（lm_head 与 embed_tokens 权重 tie，已包含在 thinker 参数内）；
    # Talker 不参与，语音能力保持原样。
    train_params = list(model.thinker.parameters())
    n_train = sum(p.numel() for p in train_params)
    print(f"[student] 可训练参数(Thinker) = {n_train/1e6:.2f}M")
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    gstep = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        for step, (x, y) in enumerate(loader):
            x, y = x.to(args.device), y.to(args.device)
            # 只走 Thinker 文本路径：embed->8 层->norm，再到 lm_head，绕过 Talker
            h, _, aux = model.thinker(x)
            logits = model.lm_head(h)
            x_logit, y_lab = logits[:, :-1, :].contiguous(), y[:, 1:].contiguous()
            loss = ce(x_logit.view(-1, x_logit.size(-1)), y_lab.view(-1))
            # Dense 无 aux；若学生为 MoE 才会有 aux（本脚本目标 Dense）
            loss = loss / args.accum
            loss.backward()
            if (step + 1) % args.accum == 0:
                nn.utils.clip_grad_norm_(train_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch+1} step {step}/{len(loader)} loss {loss.item()*args.accum:.4f}")
        print(f"[student] epoch {epoch+1} 用时 {time.time()-t0:.0f}s")

    # ===== 保存：既写 HF 目录（save_pretrained），也留一份 state_dict 兜底 =====
    model.eval()
    state_path = os.path.join(args.out_dir, "thinker_sft_state.pth")
    torch.save(model.state_dict(), state_path)
    print(f"[student] state_dict 已保存：{state_path}")
    try:
        model.save_pretrained(args.out_dir)
        tok.save_pretrained(args.out_dir)
        print(f"[student] HF 模型目录已保存：{args.out_dir}（可直接 from_pretrained 加载）")
    except Exception as e:
        print(f"[student] save_pretrained 失败（可改用 state 加载）：{e}")


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
