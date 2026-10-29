"""
kd_eval.py —— 序列级 KD 第 3 步：评测蒸馏前后的学生是否更接近教师

逐个加载模型（避免 CPU 同时占内存），对同一批问题贪心生成，比较：
  - 教师（MoE）           作为参照（soft target 的来源）
  - 基线学生（原 Dense）    蒸馏前
  - 蒸馏学生（kd_dense）    蒸馏后
关键指标：与教师的“平均字符相似度”，蒸馏后应上升；完全一致率也应上升。

CPU 冒烟：
  python kd_eval.py --teacher_dir ./minimind-3o-moe --base_dir ./minimind-3o \
      --student_dir ./kd_dense --limit 8 --max_new_tokens 32 --device cpu
"""
import gc
import json
import time
import argparse
import difflib
import torch

from kd_common import load_model, generate_answer, load_prompts


def answer_all(model_dir, prompts, device, dtype, max_new_tokens, tag, state=""):
    print(f"\n[eval] 加载 {tag}: {model_dir}")
    model, tok = load_model(model_dir, device, dtype)
    if state:
        sd = torch.load(state, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"[eval] 已载入额外 state：{state}")
    out = {}
    for i, p in enumerate(prompts):
        st = time.time()
        ans, ntk = generate_answer(model, tok, p, device, max_new_tokens, temperature=0.0)
        out[p] = ans
        print(f"{tag} {i+1:03d}/{len(prompts)} {ntk:3d}tok {time.time()-st:5.1f}s -> {ans[:40]!r}")
    del model, tok
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", default="./minimind-3o-moe")
    ap.add_argument("--base_dir", default="./minimind-3o", help="蒸馏前的 Dense 基座")
    ap.add_argument("--student_dir", default="./kd_dense", help="蒸馏后的学生目录")
    ap.add_argument("--student_state", default="", help="可选：若只有 state.pth 则在此叠加")
    ap.add_argument("--prompts", default="")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--out", default="kd_eval.json")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts, args.limit)
    teacher = answer_all(args.teacher_dir, prompts, args.device, args.dtype,
                         args.max_new_tokens, "TEACHER")
    base = answer_all(args.base_dir, prompts, args.device, args.dtype,
                      args.max_new_tokens, "BASE")
    student = answer_all(args.student_dir, prompts, args.device, args.dtype,
                         args.max_new_tokens, "STUDENT", args.student_state)

    rows = []
    exact_b = exact_s = 0
    sim_b = sim_s = 0.0
    n = len(prompts)
    for p in prompts:
        t, b, s = teacher[p], base[p], student[p]
        sb, ss = sim(b, t), sim(s, t)
        sim_b += sb; sim_s += ss
        exact_b += int(b == t); exact_s += int(s == t)
        rows.append({"prompt": p, "teacher": t, "base": b, "student": s,
                     "sim_base": round(sb, 3), "sim_student": round(ss, 3)})

    summary = {
        "n": n,
        "sim_base_to_teacher": round(sim_b / n, 4),
        "sim_student_to_teacher": round(sim_s / n, 4),
        "exact_base": f"{exact_b}/{n}",
        "exact_student": f"{exact_s}/{n}",
    }
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n================ 蒸馏评测（相对教师） ================")
    print(f"基线学生 -> 教师 相似度 : {summary['sim_base_to_teacher']}")
    print(f"蒸馏学生 -> 教师 相似度 : {summary['sim_student_to_teacher']}  （应上升）")
    print(f"完全一致：基线 {exact_b}/{n}  ->  蒸馏 {exact_s}/{n}")
    print(f"明细已写出：{args.out}")


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
