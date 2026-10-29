"""
kd_train_logit.py —— Logit 级（Hinton 软标签）知识蒸馏：MoE 教师 -> Dense 学生

和序列级（kd_train_student.py）的区别：
  - 序列级：教师只给"最终整句文本"（硬标签，one-hot），学生做普通 SFT。
  - Logit 级：训练时【教师在线】，对每个 token 输出整张词表概率分布（软标签），
    学生用 KL 散度去逼近这张分布，外加一个硬标签 CE（Hinton 标准损失）。

为什么教师/学生结构不同也能蒸馏：
  - 教师 MoE 315M、学生 Dense 113M，内部层数专家不同无所谓；
  - 两者的 lm_head 都输出同一词表 V=6400 的 logits，且用同一个 tokenizer；
  - 我们只在【输出概率分布】这一层对齐：都是 6400 个坐标，坐标含义一致，
    因此可以逐坐标做 KL。内部 hidden 不同不需要对齐（那是特征层蒸馏，本脚本不做）。

数据没变：仍读 kd_sft_lamp.jsonl（token id + 回答段 mask）。"多出来的概率分布"
  不存文件，而是每个 batch 用教师现场 forward 算出来（在线软标签）。

GPU 推荐（教师+学生同时在显存）：
  python kd_train_logit.py --teacher_dir ./minimind-3o-moe --student_dir ./minimind-3o \
      --data kd_sft_lamp.jsonl --out_dir ./kd_dense_logit \
      --epochs 3 --batch_size 4 --temperature 4.0 --alpha 0.7 --device cuda
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kd_common import load_model
from kd_train_student import TextSFTJsonl, collate  # 复用同一套数据封装/padding，不重复造轮子


def soft_kl_loss(s_logits, t_logits, labels, T):
    """Hinton 软标签损失（只在回答段、用下一个 token 对齐）。

    s_logits/t_logits: [B, T, V] 学生/教师对每个位置的 logits（V=6400）。
    labels:            [B, T]    -100 表示不计算（问题段/pad）。
    返回 T^2 缩放后的平均 KL（每个有效 token 一份）。
    """
    sl = s_logits[:, :-1, :]                 # 学生在 0..T-2 位置的预测
    tl = t_logits[:, :-1, :]                 # 教师在同样位置的预测（目标分布）
    y = labels[:, 1:]                        # 对应标签位置 1..T-1
    mask = (y != -100).float()               # 只在回答段（含 eos）上对齐，问题段/pad 为 0

    log_p_s = F.log_softmax(sl / T, dim=-1)  # 学生 log 概率，温度 T 软化
    p_t = F.softmax(tl / T, dim=-1)          # 教师软目标分布，温度 T 软化
    # KL(p_t || p_s)，先在词表维求和，得到每个位置的 KL[B,T-1]
    kl = F.kl_div(log_p_s, p_t, reduction="none").sum(-1)
    kl = (kl * mask).sum() / mask.sum().clamp(min=1.0)  # 只对回答段取平均
    return kl * (T * T)                      # Hinton 标准：T^2 补偿温度对梯度尺度的影响


def hard_ce_loss(s_logits, labels, vocab):
    """普通硬标签交叉熵（one-hot 正确答案），和序列级完全一致。"""
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    sl = s_logits[:, :-1, :].reshape(-1, vocab)
    y = labels[:, 1:].reshape(-1)
    return ce(sl, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", default="./minimind-3o-moe", help="教师（MoE）")
    ap.add_argument("--student_dir", default="./minimind-3o", help="学生基座（Dense）")
    ap.add_argument("--data", default="kd_sft_lamp.jsonl")
    ap.add_argument("--out_dir", default="./kd_dense_logit")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=4.0, help="温度 T，常用 2~4")
    ap.add_argument("--alpha", type=float, default=0.7, help="软标签权重；硬标签权重=1-alpha")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--student_dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--teacher_dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 加载教师：冻结、eval、不反传；用 bf16 省显存（logits 算 KL 时再转 fp32） ----
    teacher, tok = load_model(args.teacher_dir, args.device, args.teacher_dtype)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---- 加载学生：fp32 训练（梯度需要精度） ----
    student, _ = load_model(args.student_dir, args.device, args.student_dtype)
    student.train()

    ds = TextSFTJsonl(args.data, tok, args.max_seq_len)
    if args.limit:
        ds.samples = ds.samples[: args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    print(f"[logit-kd] 样本={len(ds)} 批次={len(loader)} T={args.temperature} alpha={args.alpha}")

    train_params = list(student.thinker.parameters())   # 只训练学生 Thinker（lm_head 与 embed tie）
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)

    for epoch in range(args.epochs):
        for step, (x, y) in enumerate(loader):
            x, y = x.to(args.device), y.to(args.device)

            # 教师在线前向：只算、不更新，输出整张词表分布作为软目标
            with torch.no_grad():
                th, _, _ = teacher.thinker(x)
                t_logits = teacher.lm_head(th).detach().float()   # [B,T,V] 固定不变

            # 学生前向
            sh, _, _ = student.thinker(x)
            s_logits = student.lm_head(sh)                        # [B,T,V] 会被更新
            V = s_logits.size(-1)

            loss_soft = soft_kl_loss(s_logits, t_logits, y, args.temperature)  # 软标签 KL
            loss_hard = hard_ce_loss(s_logits, y, V)                            # 硬标签 CE
            loss = args.alpha * loss_soft + (1.0 - args.alpha) * loss_hard      # Hinton 总损失
            (loss / args.accum).backward()                                     # 只更新学生

            if (step + 1) % args.accum == 0:
                nn.utils.clip_grad_norm_(train_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                print(f"epoch {epoch+1} step {step}/{len(loader)} "
                      f"loss {loss.item():.4f} (soft {loss_soft.item():.4f} hard {loss_hard.item():.4f})")

    # ---- 保存：和序列级完全一样的 HF 目录 + state 兜底 ----
    student.eval()
    torch.save(student.state_dict(), os.path.join(args.out_dir, "thinker_sft_state.pth"))
    try:
        student.save_pretrained(args.out_dir)
        tok.save_pretrained(args.out_dir)
        print(f"[logit-kd] 已保存：{args.out_dir}（可用 kd_eval_intent.py 评测，无需改）")
    except Exception as e:
        print(f"[logit-kd] save_pretrained 失败，可用 state 兜底：{e}")


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
