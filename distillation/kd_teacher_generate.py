"""
kd_teacher_generate.py —— 序列级 KD 第 1 步：让教师（MoE）批量生成答案，产出学生 SFT 数据集

输出：一个 JSONL，每行一个 {"conversations":[{user},{assistant}]}，
      与 minimind 的 SFTDataset / apply_chat_template 格式一致，可直接喂给 kd_train_student.py。

CPU 冒烟（笔记本，慢，先验证流程）：
  python kd_teacher_generate.py --teacher_dir ./minimind-3o-moe --limit 8 --max_new_tokens 24 --device cpu
GPU/云端全量：
  python kd_teacher_generate.py --teacher_dir ./minimind-3o-moe --prompts my_prompts.jsonl \
      --limit 8000 --max_new_tokens 96 --device cuda --dtype fp32
"""
import os
import json
import time
import argparse
import torch

from kd_common import load_model, generate_answer, load_prompts, make_sft_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", default="./minimind-3o-moe", help="教师模型目录（默认 MoE）")
    ap.add_argument("--prompts", default="", help="prompt 文件（jsonl/txt/json）；不传则用内置 50 问")
    ap.add_argument("--out", default="kd_sft.jsonl", help="输出 SFT jsonl 路径")
    ap.add_argument("--limit", type=int, default=50, help="生成多少条（0=全部）")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0, help="0=贪心，教师标签可复现")
    ap.add_argument("--drop_truncated", action="store_true",
                    help="丢弃撞到 max_new_tokens 上限、没有 EOS 的答案（截断/没收尾）")
    ap.add_argument("--min_answer_len", type=int, default=1,
                    help="答案至少多少字符；空/过短的丢弃（默认>=1）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="1070(Pascal)用fp32；4080可用bf16加速")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts, args.limit)
    print(f"[teacher] 设备={args.device} dtype={args.dtype} 待生成 prompt={len(prompts)} 条")
    model, tok = load_model(args.teacher_dir, args.device, args.dtype)

    # 断点续跑：已经存在的输出读出来，跳过已完成的 prompt
    done = {}
    if os.path.exists(args.out):
        for line in open(args.out, "r", encoding="utf-8"):
            try:
                r = json.loads(line)
                p = r["conversations"][0]["content"]
                done[p] = 1
            except Exception:
                pass
    print(f"[teacher] 已有 {len(done)} 条结果，将跳过")

    n_new = n_drop = 0
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            if p in done:
                continue
            st = time.time()
            try:
                ans, ntk = generate_answer(
                    model, tok, p, args.device,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            except Exception as e:
                print(f"[teacher] 生成失败，跳过：{p[:30]} | {e}")
                continue
            # —— 数据质量过滤：教师是上限，坏标签不要进训练集 ——
            if len(ans.strip()) < args.min_answer_len:
                n_drop += 1
                print(f"{i+1:04d}/{len(prompts)} DROP(empty) {ntk:3d}tok | {p[:30]}")
                continue
            if args.drop_truncated and ntk >= args.max_new_tokens:
                # 顶到上限说明没有 EOS：答案被截断/没收尾，学生会学成长句不停
                n_drop += 1
                print(f"{i+1:04d}/{len(prompts)} DROP(trunc) {ntk:3d}tok | {p[:30]} -> {ans[:40]!r}")
                continue
            rec = make_sft_record(p, ans)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            print(f"{i+1:04d}/{len(prompts)} {ntk:3d}tok {time.time()-st:5.1f}s | "
                  f"{p[:30]} -> {ans[:50]!r}")

    total = sum(1 for _ in open(args.out, encoding="utf-8"))
    print(f"[teacher] 完成：新增 {n_new} 条，丢弃 {n_drop} 条，输出文件 {args.out}，总耗时 {time.time()-t0:.0f}s")
    print(f"[teacher] 文件总行数 = {total}")


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
