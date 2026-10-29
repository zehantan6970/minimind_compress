#!/usr/bin/env python3
"""
main.py - thin convenience launcher for the MiniMind compression / KD project.

It does NOT reimplement any training logic. It simply builds the command for the
existing standalone scripts (quantization/bench_quant_omni.py and the kd_* scripts
in distillation/) and runs them with sensible defaults, so you don't have to
remember every flag.

Examples (single lines):
    python main.py make-data
    python main.py train-hard
    python main.py train-logit
    python main.py eval
    python main.py infer --prompt "Turn on the light."
    python main.py quant
    python main.py --dry-run train-logit          # only print the command
    python main.py --device cpu --scripts-dir ./scripts quant

You can also pass extra arguments straight through to the underlying script:
    python main.py quant -- --scope ffn
"""
import argparse
import os
import subprocess
import sys

# ---- Defaults (override with CLI flags) ------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEACHER = "./minimind-3o-moe"
DEFAULT_BASE = "./minimind-3o"
DEFAULT_STUDENT = "./kd_dense_lamp2"
DEFAULT_DATA = "kd_sft_lamp.jsonl"
DEFAULT_EVAL = "kd_eval_intents.jsonl"
DEFAULT_DEVICE = "cuda"


def find_script(name: str, scripts_dir: str | None) -> str:
    """Locate a script: explicit --scripts-dir, then distillation/quantization."""
    candidates = []
    if scripts_dir:
        candidates.append(os.path.join(scripts_dir, name))
    if name.startswith("bench_"):
        candidates.append(os.path.join(HERE, "quantization", name))
    else:
        candidates.append(os.path.join(HERE, "distillation", name))
    for path in candidates:
        if os.path.isfile(path):
            return path
    sys.exit(f"[main] cannot find '{name}'. Pass --scripts-dir or check the repo layout.")


def run(cmd: list[str], dry_run: bool) -> None:
    print("[main] " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMind compression / KD launcher")
    parser.add_argument("--scripts-dir", default=None,
                        help="folder holding the scripts (e.g. MiniMind's scripts/)")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("make-data", help="generate the 211 template examples")
    sub.add_parser("train-hard", help="student hard-label SFT (best command model)")
    sub.add_parser("train-logit", help="logit-level soft-label KD")
    sub.add_parser("eval", help="intent-accuracy evaluation")
    sub.add_parser("quant", help="weight-only INT8 benchmark")
    p_infer = sub.add_parser("infer", help="deployment inference")
    p_infer.add_argument("--prompt", default=None)
    p_dump = sub.add_parser("dump", help="real 6400-dim diagnostic")
    p_dump.add_argument("--prompt", default="Turn on the light.")
    p_dump.add_argument("--answer", default="Sure, I'll turn on the light.")
    p_dump.add_argument("--target-word", default="on")

    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    sd = args.scripts_dir

    if args.action == "make-data":
        run([sys.executable, find_script("kd_make_synthetic.py", sd),
             "--out", args.data] + extra, args.dry_run)

    elif args.action == "train-hard":
        run([sys.executable, find_script("kd_train_student.py", sd),
             "--student_dir", args.base, "--data", args.data,
             "--out_dir", args.student, "--epochs", "3", "--batch_size", "8",
             "--lr", "1e-4", "--max_seq_len", "384", "--device", args.device] + extra,
            args.dry_run)

    elif args.action == "train-logit":
        run([sys.executable, find_script("kd_train_logit.py", sd),
             "--teacher_dir", args.teacher, "--student_dir", args.base,
             "--data", args.data, "--out_dir", "./kd_dense_logit",
             "--epochs", "3", "--batch_size", "4", "--temperature", "4.0",
             "--alpha", "0.3", "--device", args.device] + extra, args.dry_run)

    elif args.action == "eval":
        run([sys.executable, find_script("kd_eval_intent.py", sd),
             "--base_dir", args.base, "--student_dir", args.student,
             "--prompts", DEFAULT_EVAL, "--max_new_tokens", "128",
             "--device", args.device] + extra, args.dry_run)

    elif args.action == "quant":
        run([sys.executable, find_script("bench_quant_omni.py", sd),
             "--model_dir", args.base, "--limit", "50", "--scope", "blocks"] + extra,
            args.dry_run)

    elif args.action == "infer":
        cmd = [sys.executable, find_script("kd_infer.py", sd),
               "--model_dir", args.student, "--device", args.device]
        if args.prompt:
            cmd += ["--prompt", args.prompt]
        run(cmd + extra, args.dry_run)

    elif args.action == "dump":
        run([sys.executable, find_script("kd_dump_logits.py", sd),
             "--teacher_dir", args.teacher, "--base_dir", args.base,
             "--student_dir", "./kd_dense_logit", "--prompt", args.prompt,
             "--answer", args.answer, "--target_word", args.target_word,
             "--temperature", "4.0", "--device", args.device] + extra, args.dry_run)


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
