"""
kd_eval_intent.py —— 方案 C（混合标签）的正确评测：意图正确率

为什么不能再用 kd_eval.py 的"相似度"：
  kd_eval 把教师答案当唯一答案键，比的是 base/student 与教师的文本相似度。
  但在混合标签下，指令类（开灯/关灯…）的正确答案是【确定性模板】，而 MoE 教师对
  指令只会打太极（"It seems like you're referring to..."）。学生答对模板时，反而和
  教师不像，所以相似度会误导。
  本脚本改为【规则意图判定】：对每条留出 prompt 标注期望 intent，用该意图的
  关键词短语去匹配学生回答，统计 base 与 student 的意图正确率（这才是产品要的指标）。

用法：
  python kd_eval_intent.py --base_dir ./minimind-3o --student_dir ./kd_dense_lamp \
      --prompts kd_eval_intents.jsonl --max_new_tokens 128 --device cuda
"""
import os
import json
import gc
import argparse
import torch

from kd_common import load_model, generate_answer


# 每个意图可接受的短语（小写子串匹配，命中任意一个即算答对）。
# 短语取自模板回复的核心词，学生即使换了句式也应包含这些词。
ACCEPT = {
    "turn_on":       ["turn on", "turning the light on", "light is on", "on it"],
    "turn_off":      ["turn off", "turning the light off", "light is off", "off it goes"],
    "brighter":      ["brighter", "increasing the brightness", "making it brighter", "more light"],
    "dimmer":        ["dim", "softer", "lowering the brightness", "less light"],
    "scene_warm":    ["warm", "cozy", "reading mode"],
    "nod":           ["nod"],
    "stop":          ["stop", "stopped", "stopping"],
    "repeat":        ["again", "repeat", "say it once more"],
    "speak_slower":  ["slow"],
    "speak_louder":  ["louder", "volume", "speak up"],
    "refuse":        ["can't", "cannot", "sorry", "beyond", "just a smart lamp", "i'm a lamp"],
    "identity":      ["smart lamp", "desk lamp", "i can", "i'm a", "i am"],
    "greeting":      ["hi", "hello", "hey", "morning", "afternoon", "evening"],
    "thanks":        ["welcome", "anytime", "happy to help"],
    "goodbye":       ["bye", "goodbye", "see you", "later", "good night"],
}


def load_intent_prompts(path, limit=0):
    """读取留出评测：每行 {"prompt": "...", "intent": "turn_on"}（可带 "accept" 覆盖）。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:limit] if limit and limit > 0 else items


def is_correct(intent, answer, accept=None):
    """回答中包含该意图任一可接受短语（忽略大小写）即算意图正确。"""
    phrases = accept or ACCEPT.get(intent, [])
    low = answer.lower()
    return any(p.lower() in low for p in phrases)


def run_model(model_dir, items, device, dtype, max_new_tokens):
    """加载一个模型，对全部留出题贪心作答并判意图，返回逐条结果与正确率。"""
    model, tok = load_model(model_dir, device, dtype)
    rows, n_ok = [], 0
    for i, it in enumerate(items):
        ans, ntk = generate_answer(model, tok, it["prompt"], device,
                                   max_new_tokens=max_new_tokens, temperature=0.0)
        ok = is_correct(it["intent"], ans, it.get("accept"))
        n_ok += int(ok)
        rows.append({"prompt": it["prompt"], "intent": it["intent"],
                     "answer": ans, "ntok": ntk, "ok": ok})
        flag = "OK " if ok else "XX "
        print(f"  {flag}{i+1:03d}/{len(items)} [{it['intent']:13s}] {it['prompt'][:34]:34s} -> {ans[:46]}")
    # 卸载显存，给下一个模型腾地方
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, n_ok / max(1, len(items))


def per_intent(rows):
    stat = {}
    for r in rows:
        d = stat.setdefault(r["intent"], [0, 0])
        d[1] += 1
        d[0] += int(r["ok"])
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="./minimind-3o", help="训练前基座（对照）")
    ap.add_argument("--student_dir", default="./kd_dense_lamp", help="蒸馏后学生")
    ap.add_argument("--prompts", default="kd_eval_intents.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--out", default="kd_eval_intent.json")
    args = ap.parse_args()

    items = load_intent_prompts(args.prompts, args.limit)
    print(f"[intent-eval] 留出题 {len(items)} 条，设备 {args.device}")

    print(f"[intent-eval] 加载 BASE: {args.base_dir}")
    base_rows, base_acc = run_model(args.base_dir, items, args.device, args.dtype, args.max_new_tokens)
    print(f"[intent-eval] 加载 STUDENT: {args.student_dir}")
    stu_rows, stu_acc = run_model(args.student_dir, items, args.device, args.dtype, args.max_new_tokens)

    bstat, sstat = per_intent(base_rows), per_intent(stu_rows)
    print("\n========== 意图正确率（base -> student） ==========")
    print(f"{'intent':14s} {'base':>10s} {'student':>10s}")
    for intent in sorted(set(bstat) | set(sstat)):
        b, s = bstat.get(intent, [0, 0]), sstat.get(intent, [0, 0])
        print(f"{intent:14s} {b[0]:4d}/{b[1]:<4d} {s[0]:4d}/{s[1]:<4d}")
    print("-" * 40)
    print(f"{'TOTAL':14s} {base_acc*100:7.1f}%  {stu_acc*100:7.1f}%")
    print("（学生意图正确率应显著高于 base；指令类这才是真正的成功指标）")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"base": base_rows, "student": stu_rows,
                   "base_acc": base_acc, "student_acc": stu_acc},
                  f, ensure_ascii=False, indent=2)
    print(f"明细已写出：{args.out}")


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
