# Weight-only INT8 Quantization Benchmark

This folder contains `bench_quant_omni.py`, a self-contained benchmark that compares the **FP32 baseline** and **Weight-only INT8** versions of a MiniMind-O model on the same fixed questions (greedy decoding).

> 中文说明见根目录 [readme_cn.md](../readme_cn.md)。

## Why this approach

- Tools like bitsandbytes / GPTQ / AWQ rely on newer kernels that do **not** run on older GPUs such as Pascal (GTX1070, sm_61), which has no INT8 Tensor Cores.
- We therefore use a custom **`W8Linear`**: weights are stored as **8-bit**, and at the forward pass they are **de-quantized back to FP32** for an ordinary matmul. This lowers storage / memory bandwidth and runs on **any device, including CPU**.

### Linear weight quantization (per-channel)

```
scale   = max(|w|) / 127
w_q     = clamp( round(w / scale), -127, 127 )   # stored as int8
w_hat   = w_q * scale                            # de-quantized at forward
```

The parameter count is unchanged; only the storage precision changes (16/32-bit → 8-bit).

## Files

| File | Purpose |
|---|---|
| `bench_quant_omni.py` | Replaces `nn.Linear` with `W8Linear`, runs FP32 vs INT8, reports size / memory / speed / answer consistency, writes `quant_report.md` and `.csv`. |

## Setup

Place the script in MiniMind's `minimind-o/scripts/` directory (next to `minimind-3o/` and optionally `minimind-3o-moe/`) and **run it from `scripts/`**. Running from there avoids loading SenseVoice / SigLIP / Mimi.

```bash
# Smoke test (8 questions)
python bench_quant_omni.py --model_dir ./minimind-3o --limit 8

# Full 50 questions: quantize attention + FFN (maximum compression)
python bench_quant_omni.py --model_dir ./minimind-3o --limit 50 --scope blocks

# MoE (Router gates are automatically kept in FP32)
python bench_quant_omni.py --model_dir ./minimind-3o-moe --limit 50 --scope blocks

# More conservative: FFN / experts only
python bench_quant_omni.py --model_dir ./minimind-3o --limit 50 --scope ffn

# Save the quantized weights
python bench_quant_omni.py --model_dir ./minimind-3o --scope blocks --save minimind3o_w8.pth
```

## How to read the report

| Metric | Expectation |
|---|---|
| Weight size | INT8 clearly lower (the decoder is the bulk, about 1/4 of FP32 storage) |
| Peak memory | INT8 lower |
| tok/s | On Pascal at least not worse (bandwidth-bound, usually slightly faster) |
| Answer consistency | Very high under greedy decoding; similarity > 0.95 is acceptable |

- If `--scope blocks` degrades, fall back to `--scope ffn`.
- Model size in this project: about **452 MB (FP32) → 207 MB (INT8)**.

## Troubleshooting

- `no kernel image is available for execution on the device`: install the cu126 wheel (cu130 drops sm_61).
- `import onnxruntime` fails: `pip install onnxruntime`.
- The script loads SenseVoice / SigLIP: you are not in `scripts/` or `./model` exists; return to `scripts/`.
#（注：内容由AI生成）
