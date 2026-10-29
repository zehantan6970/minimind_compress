"""
kd_infer.py —— 蒸馏后学生的【部署推理】脚本（加载一次，反复问答）

和训练的区别：
  - 训练：teacher-forced，整段答案喂进去，一次前向算所有位置的 loss；
  - 推理：只给 prompt，模型一个 token 一个 token 地自己往后写（自回归），
          每步把新 token 拼回输入，直到 EOS 或达到长度上限；不计算 loss、不需要教师。

用法：
  # 单条
  python kd_infer.py --model_dir ./kd_dense_lamp2 --prompt "Turn on the light." --device cuda
  # 交互模式（不传 prompt，回车输入，q 退出）
  python kd_infer.py --model_dir ./kd_dense_lamp2 --device cuda
"""
import argparse
import torch

from kd_common import load_model, generate_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./kd_dense_lamp2", help="要部署的学生目录（也可填 minimind-3o / kd_dense_logit）")
    ap.add_argument("--prompt", default="", help="单条提问；不传则进入交互模式")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    # 部署时只加载学生一个模型，教师完全不需要
    model, tok = load_model(args.model_dir, args.device, args.dtype)
    print(f"[infer] 已加载 {args.model_dir}，设备 {args.device}")

    def ask(prompt):
        ans, n = generate_answer(model, tok, prompt, args.device,
                                 max_new_tokens=args.max_new_tokens, temperature=0.0)
        print(f"USER > {prompt}")
        print(f"LAMP > {ans}   （生成 {n} 个 token）\n")

    if args.prompt:
        ask(args.prompt)
        return

    # 交互模式：模型只加载一次，后续问答复用（generate 内部走 KV cache）
    print("交互模式，输入问题回车；输入 q 退出。")
    while True:
        try:
            prompt = input("USER > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("q", "quit", "exit"):
            break
        if prompt:
            ask(prompt)


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
