# -*- coding: utf-8 -*-
"""
================================================================================
M1–M2：MiniMind-O 生成器 Weight-only INT8 量化 · 基线对比与回归脚本
文件：scripts/bench_quant_omni.py
适用：GTX 1070（Pascal sm_61，cu126）—— 用 FP32 计算 + INT8 存权重，不依赖
      bnb/GPTQ/AWQ 新内核；也能在任意 CPU/云端跑。

它做三件事：
  1) 加载同一个模型两次：一次 FP32（基线），一次把“解码层里的 Linear”换成 W8Linear；
  2) 用完全相同的固定问题（贪心解码，temperature=0）分别生成，逐题对比；
  3) 输出权重占用、显存/内存、token 速度、答案一致率，并写 Markdown + CSV 报告。

只做【文本】回归（不加载 SenseVoice/SigLIP/Mimi，最轻量），因为量化的主体是
Thinker/Talker 的权重；语音/图像在 M5 端到端联调时再验证。

运行（务必在 scripts/ 目录下，这样 ./model 找不到编码器，自动只建文本模型）：
  cd scripts
  # 先小规模冒烟：
  python bench_quant_omni.py --model_dir ./minimind-3o --limit 8
  # 没问题再跑 50 问：
  python bench_quant_omni.py --model_dir ./minimind-3o --limit 50
  # MoE 权重：
  python bench_quant_omni.py --model_dir ./minimind-3o-moe --limit 50 --scope blocks
================================================================================
"""

import os
import sys
# 保证能 import 仓库里的 model.model_omni（trust_remote_code 解析依赖）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import time
import json
import logging
import difflib
import argparse
import traceback
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# 统一 logger：控制台 + 文件双写，方便 SSH 后台跑完后查问题（见 setup_logging）
logger = logging.getLogger('bench_quant')


def setup_logging(log_path: str):
    """配置同时输出到屏幕和日志文件；带时间戳、级别。"""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)


# =============================================================================
# 一、Weight-only INT8：可移植的量化 Linear（FP32 计算，INT8 存权重）
# =============================================================================
class W8Linear(nn.Module):
    """按“输出通道”做 per-channel 对称 INT8 量化；前向时反量化为 x.dtype 再 GEMM。

    为什么这样设计：
      - GTX1070 是 Pascal(sm_61)，没有 INT8 Tensor Core，bnb/GPTQ/AWQ 的专用
        matmul 内核跑不了；
      - 但小模型是“内存带宽受限”，把权重用 int8 存储（体积 1/4），用的时候再
        反量化回 FP32 做普通矩阵乘，显存/带宽直接下降，且任何设备都能跑。
    """

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data.float()                       # [out, in]，转 FP32 统计
        scale = w.abs().max(dim=1, keepdim=True)[0]      # 每个输出通道的最大绝对值
        scale = scale.clamp_min(1e-8) / 127.0            # 量化尺度：max/127
        wq = torch.round(w / scale).clamp(-128, 127).to(torch.int8)  # 量化成 int8
        self.register_buffer('wq', wq)                   # int8 权重（1 字节/参数）
        self.register_buffer('scale', scale)             # 每通道反量化尺度（FP32）
        # 解码器层里的 Linear 基本都 bias=False，这里仍兼容有偏置的情况
        self.bias = nn.Parameter(lin.bias.data.clone()) if lin.bias is not None else None

    def forward(self, x):
        w = self.wq.to(x.dtype) * self.scale.to(x.dtype)  # 反量化：int8 -> x.dtype
        return nn.functional.linear(x, w, self.bias)      # 普通矩阵乘，计算精度不降级


def _leaf_and_path(qname: str):
    parts = qname.split('.')
    return parts[-1], parts


def should_quantize(qname: str, scope: str) -> bool:
    """决定一个 nn.Linear 是否量化。只量化“解码层 .layers. 内部”的大矩阵，
    并精确保留 MoE 路由器（属性名恰好叫 gate，而不是 gate_proj）。"""
    leaf, parts = _leaf_and_path(qname)
    if '.layers.' not in ('.' + qname + '.'):
        return False                                    # 层外：lm_head/各投影/输出头，全保留
    if 'gate' in parts:
        return False                                    # 精确跳过 MoE Router（gate 本体）
    if scope == 'ffn':
        # 只量化 FFN/专家：gate_proj(注意不是 gate)/up_proj/down_proj
        return leaf in ('gate_proj', 'up_proj', 'down_proj')
    return True                                         # scope == 'blocks'：注意力+FFN 都量化


def quantize_model(model: nn.Module, scope: str = 'blocks'):
    """把符合条件的 nn.Linear 原地替换为 W8Linear。返回替换数量。"""
    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                qname = f"{name}.{child_name}" if name else child_name
                if should_quantize(qname, scope):
                    targets.append((module, child_name, child, qname))
    for parent, child_name, lin, qname in targets:
        setattr(parent, child_name, W8Linear(lin))
    return len(targets)


# =============================================================================
# 二、模型加载（Pascal 上保持 FP32，不调用 .half()）
# =============================================================================
def load_model(model_dir: str, device: str, scope: str, do_quant: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # torch_dtype=float32：Pascal 上显式用 FP32（bf16/fp16 会被拉到 1/64 慢速）
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()                                       # 推理模式，关掉 dropout
    n_replaced = 0
    if do_quant:
        n_replaced = quantize_model(model, scope)
        logger.info(f'[quant] 替换了 {n_replaced} 个 nn.Linear -> W8Linear（scope={scope}）')
    model = model.to(device)                          # 1070 上保持 FP32（不要 .half）
    return model, tokenizer, n_replaced


# =============================================================================
# 三、文本生成（走官方 stream 接口，因为 omni 的非 stream 分支最终会返回 (None,frame)）
# =============================================================================
@torch.inference_mode()
def generate_text(model, tokenizer, question, device, max_new_tokens, temperature):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
    x = torch.tensor(tokenizer(text).data['input_ids'], dtype=torch.long, device=device)[None, ...]
    start = x.shape[1]
    last_y = None
    # stream=True 时 generate 是生成器，逐步 yield (文本token, 音频帧)
    gen = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_p=0.85, stream=True,
                         return_audio_codes=False, open_thinking=False)
    for y, _af in gen:
        if y is not None and y.numel() > 0:
            last_y = y                                 # 累积的完整生成 token
    if last_y is None:
        return '', 0
    new_tokens = int(last_y.shape[1])
    answer = tokenizer.decode(last_y[0].tolist(), skip_special_tokens=True)
    return answer.strip(), new_tokens


# =============================================================================
# 四、显存/内存与权重占用统计
# =============================================================================
def peak_memory_mb(device: str):
    if device.startswith('cuda'):
        return torch.cuda.max_memory_allocated() / 1e6
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def fp32_weight_bytes(model: nn.Module):
    """假设所有参数都是 FP32(4 字节) 时的权重体积（基线）。"""
    return sum(p.numel() for p in model.parameters()) * 4


def actual_weight_bytes(model: nn.Module):
    """量化后真实占用：直接按 state_dict 里每个张量的 dtype 统计字节数。
    替换后 W8Linear 的 wq(int8=1字节)+scale(FP32) 已在 state_dict 中；
    非持久化的 RoPE 缓存（freqs_cos/sin）默认不计入，正好不统计。"""
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


# =============================================================================
# 五、固定 50 问（短问题，适合 0.2B 小模型）
# =============================================================================
DEFAULT_QUESTIONS = [
    "Please introduce yourself.",
    "Tell me a joke.",
    "How do I make a cup of coffee?",
    "What's the weather like today?",
    "Will it rain tomorrow?",
    "Can you sing a song for me?",
    "What can you do?",
    "Hello, how are you?",
    "Thank you very much.",
    "Goodbye!",
    "What is your name?",
    "What time is it?",
    "Can you help me with math?",
    "What is 2 plus 3?",
    "What is 10 minus 4?",
    "What is the capital of China?",
    "Name one fruit that is red.",
    "What color is the sky?",
    "How many days are in a week?",
    "What day comes after Monday?",
    "Translate 'hello' to Chinese.",
    "Say something nice to a friend.",
    "Give me a study tip.",
    "How can I sleep better?",
    "What should I eat for breakfast?",
    "Give me a simple recipe.",
    "How do I stay healthy?",
    "Why do we need to drink water?",
    "What is a good hobby?",
    "Recommend a book for beginners.",
    "How do I learn English?",
    "How can I focus while studying?",
    "What makes a good friend?",
    "Give me a morning greeting.",
    "Cheer me up, please.",
    "What is artificial intelligence?",
    "What is a robot?",
    "Can robots replace humans?",
    "Write a short poem about the sun.",
    "Write a one-line slogan for a lamp.",
    "Turn on the light.",
    "Make the lamp brighter.",
    "Turn off the light.",
    "Nod the lamp.",
    "What is the smart lamp for?",
    "I can't hear you.",
    "Please repeat that.",
    "Speak slower, please.",
    "Stop.",
    "That's all, thanks.",
]


# =============================================================================
# 六、跑一轮（FP32 或 INT8），返回每题结果与汇总指标
# =============================================================================
def run_round(tag, model, tokenizer, questions, device, max_new_tokens, temperature):
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    rows = []
    total_new_tokens = 0
    failures = 0
    t0 = time.time()
    for i, q in enumerate(questions):
        # 第 1 题作为 warmup（不计时），让 cudnn/内核完成预热；单题异常不中断整轮
        try:
            if i == 0:
                generate_text(model, tokenizer, q, device, max_new_tokens, temperature)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            st = time.time()
            ans, ntk = generate_text(model, tokenizer, q, device, max_new_tokens, temperature)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            dt = time.time() - st
        except Exception as e:                      # 单题失败记录后继续，避免 50 问前功尽弃
            logger.error(f'[{tag}] 第{i+1}题生成失败: {e}')
            logger.error(traceback.format_exc())
            ans, ntk, dt, fail = '<ERROR>', 0, 0.0, True
            failures += 1
        total_new_tokens += ntk
        speed = (ntk / dt) if dt > 0 else 0
        rows.append({'tag': tag, 'q': q, 'answer': ans, 'new_tokens': ntk,
                     'time_s': round(dt, 3), 'tok_s': round(speed, 2)})
        logger.info(f'[{tag}] {i+1:02d}/{len(questions)} {ntk:3d}tok {speed:5.1f}tok/s '
                    f'| {q[:28]} -> {ans[:40]!r}')
    total_time = time.time() - t0
    mem = peak_memory_mb(device)
    summary = {
        'tag': tag,
        'total_time_s': round(total_time, 2),
        'avg_tok_s': round(total_new_tokens / max(total_time, 1e-9), 2),
        'peak_mem_mb': round(mem, 1),
        'failures': failures,
    }
    return rows, summary


# =============================================================================
# 七、对比 + 报告
# =============================================================================
def compare_and_report(base_rows, w8_rows, base_sum, w8_sum, base_bytes, w8_bytes,
                       n_replaced, out_prefix):
    assert len(base_rows) == len(w8_rows)
    exact = 0
    sim_sum = 0.0
    n = len(base_rows)
    detail = []
    for b, q in zip(base_rows, w8_rows):
        a, c = b['answer'], q['answer']
        is_exact = (a == c)
        exact += int(is_exact)
        sim = difflib.SequenceMatcher(None, a, c).ratio()   # 字符级相似度，0~1
        sim_sum += sim
        detail.append((b['q'], a, c, is_exact, round(sim, 3)))

    logger.info('================ 量化对比结果 ================')
    logger.info(f'FP32 权重体积 : {base_bytes/1e6:8.1f} MB')
    logger.info(f'INT8 权重体积 : {w8_bytes/1e6:8.1f} MB  (压缩到 {w8_bytes/base_bytes*100:.1f}%)')
    logger.info(f'量化 Linear 数 : {n_replaced}')
    logger.info(f'峰值内存/显存 : FP32 {base_sum["peak_mem_mb"]}MB  ->  INT8 {w8_sum["peak_mem_mb"]}MB')
    logger.info(f'平均速度       : FP32 {base_sum["avg_tok_s"]} tok/s  ->  INT8 {w8_sum["avg_tok_s"]} tok/s')
    logger.info(f'失败题数       : FP32 {base_sum["failures"]}  ->  INT8 {w8_sum["failures"]}')
    logger.info(f'答案完全一致   : {exact}/{n}  ({exact/n*100:.1f}%)')
    logger.info(f'平均字符相似度 : {sim_sum/n:.3f}')

    # Markdown 报告
    md = out_prefix + '.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# MiniMind-O Weight-only INT8 量化报告\n\n')
        f.write('| 指标 | FP32 基线 | INT8 权重 |\n|---|---|---|\n')
        f.write(f'| 权重体积(MB) | {base_bytes/1e6:.1f} | {w8_bytes/1e6:.1f} |\n')
        f.write(f'| 峰值内存/显存(MB) | {base_sum["peak_mem_mb"]} | {w8_sum["peak_mem_mb"]} |\n')
        f.write(f'| 平均速度(tok/s) | {base_sum["avg_tok_s"]} | {w8_sum["avg_tok_s"]} |\n')
        f.write(f'| 总耗时(s) | {base_sum["total_time_s"]} | {w8_sum["total_time_s"]} |\n\n')
        f.write(f'- 量化 Linear 数量：{n_replaced}\n')
        f.write(f'- 答案完全一致：{exact}/{n} ({exact/n*100:.1f}%)\n')
        f.write(f'- 平均字符相似度：{sim_sum/n:.3f}\n\n')
        f.write('## 逐题对比\n\n| # | 问题 | 完全一致 | 相似度 | FP32 答案 | INT8 答案 |\n|---|---|---|---|---|---|\n')
        for i, (qq, a, c, ie, sim) in enumerate(detail, 1):
            f.write(f'| {i} | {qq} | {"✔" if ie else "�—"} | {sim} | {a[:60]} | {c[:60]} |\n')
    # CSV 明细
    csvp = out_prefix + '.csv'
    with open(csvp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['question', 'exact', 'similarity', 'fp32_answer', 'w8_answer',
                    'fp32_tok_s', 'w8_tok_s'])
        for i, (qq, a, c, ie, sim) in enumerate(detail):
            w.writerow([qq, ie, sim, a, c, base_rows[i]['tok_s'], w8_rows[i]['tok_s']])
    logger.info(f'报告已写出：{md} 与 {csvp}')


# =============================================================================
# 主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='./minimind-3o', help='HF 模型目录（Dense 或 MoE）')
    ap.add_argument('--scope', default='blocks', choices=['blocks', 'ffn'],
                    help='blocks=量化解码层注意力+FFN；ffn=仅量化FFN/专家（最保守）')
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='0=贪心，保证 FP32 与 INT8 可复现对比')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='quant_report')
    ap.add_argument('--save', default='', help='可选：保存量化后 state_dict 到该路径')
    args = ap.parse_args()

    setup_logging(args.out + '.log')
    questions = DEFAULT_QUESTIONS[:args.limit]
    logger.info('='*60)
    logger.info(f'torch={torch.__version__}  device={args.device}')
    if args.device.startswith('cuda'):
        logger.info(f'GPU={torch.cuda.get_device_name(0)}')
        logger.info(f'arch_list={torch.cuda.get_arch_list()}')
    logger.info(f'model_dir={args.model_dir}  scope={args.scope}  '
                f'问题数={len(questions)}  max_new_tokens={args.max_new_tokens}  '
                f'temperature={args.temperature}')
    logger.info('='*60)

    # ---- FP32 基线 ----
    logger.info('[1/2] 加载 FP32 基线模型 …')
    base_model, tok, _ = load_model(args.model_dir, args.device, args.scope, do_quant=False)
    base_bytes = fp32_weight_bytes(base_model)
    base_rows, base_sum = run_round('FP32', base_model, tok, questions, args.device,
                                    args.max_new_tokens, args.temperature)
    with open(args.out + '.fp32.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': base_sum, 'rows': base_rows}, f, ensure_ascii=False, indent=2)
    del base_model
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    # ---- INT8 权重 ----
    logger.info('[2/2] 加载并量化（W8Linear）…')
    w8_model, tok, n_replaced = load_model(args.model_dir, args.device, args.scope, do_quant=True)
    w8_bytes = actual_weight_bytes(w8_model)
    w8_rows, w8_sum = run_round('INT8', w8_model, tok, questions, args.device,
                                args.max_new_tokens, args.temperature)
    with open(args.out + '.w8.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': w8_sum, 'rows': w8_rows}, f, ensure_ascii=False, indent=2)
    if args.save:
        torch.save(w8_model.state_dict(), args.save)
        logger.info(f'量化 state_dict 已保存：{args.save}（重载前需先 quantize_model 再 load_state_dict）')

    compare_and_report(base_rows, w8_rows, base_sum, w8_sum, base_bytes, w8_bytes,
                       n_replaced, args.out)


if __name__ == '__main__':
    main()
#（注：内容由AI生成）
