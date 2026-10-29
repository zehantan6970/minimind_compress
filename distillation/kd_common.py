"""
kd_common.py —— 方案 A（序列级知识蒸馏 MoE 教师 -> Dense 学生）的公共工具

放置：和 kd_teacher_generate.py / kd_train_student.py / kd_eval.py 一起放在
      minimind-o/scripts/ 目录下运行（与 bench_quant_omni.py 相同）。

作用：
  1) 加载 minimind-o 的 HF 模型（教师 MoE / 学生 Dense）；
  2) 用官方 stream generate 做“纯文本贪心生成”（与量化基准完全一致，不生成音频）；
  3) 内置/读取教师 prompt 列表。
"""
import os
import sys
import json
import torch

# 让脚本能 import 仓库里的 model.model_omni（trust_remote_code 解析依赖）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModelForCausalLM


# 内置 50 个英文短问题（和 bench_quant_omni.py 一致），CPU 冒烟时不用准备外部数据
BUILTIN_PROMPTS = [
    "Please introduce yourself.", "Tell me a joke.", "How do I make a cup of coffee?",
    "What's the weather like today?", "Will it rain tomorrow?", "Can you sing a song for me?",
    "What can you do?", "Hello, how are you?", "Thank you very much.", "Goodbye!",
    "What is your name?", "What time is it?", "Can you help me with math?",
    "What is 2 plus 3?", "What is 10 minus 4?", "What is the capital of China?",
    "Name one fruit that is red.", "What color is the sky?", "How many days are in a week?",
    "What day comes after Monday?", "Translate 'hello' to Chinese.", "Say something nice to a friend.",
    "Give me a study tip.", "How can I sleep better?", "What should I eat for breakfast?",
    "Give me a simple recipe.", "How do I stay healthy?", "Why do we need to drink water?",
    "What is a good hobby?", "Recommend a book for beginners.", "How do I learn English?",
    "How can I focus while studying?", "What makes a good friend?", "Give me a morning greeting.",
    "Cheer me up, please.", "What is artificial intelligence?", "What is a robot?",
    "Can robots replace humans?", "Write a short poem about the sun.",
    "Write a one-line slogan for a lamp.", "Turn on the light.", "Make the lamp brighter.",
    "Turn off the light.", "Nod the lamp.", "What is the smart lamp for?",
    "I can't hear you.", "Please repeat that.", "Speak slower, please.", "Stop.",
    "That's all, thanks.",
]


def resolve_dtype(name: str, device: str):
    """把字符串 dtype 解析成 torch dtype；CPU 一律建议 fp32（Pascal 也用 fp32）。"""
    if name == "fp16":
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def load_model(model_dir: str, device: str = "cpu", dtype: str = "fp32"):
    """加载 minimind-o 模型（trust_remote_code）。返回 (model, tokenizer)。

    从 scripts/ 目录运行时，./model 通常不存在，编码器不会加载（只保留 Thinker/Talker）。
    """
    torch_dtype = resolve_dtype(dtype, device)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tok


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, device: str,
                    max_new_tokens: int = 64, temperature: float = 0.0):
    """纯文本贪心生成（temperature=0 等价贪心），返回 (答案文本, 新token数)。

    走官方 stream=True，循环取最后一个非 None 的 y；
    避开非 stream 末尾返回 (None, audio_frame) 的坑。不生成音频。
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
    x = torch.tensor(tokenizer(text).data["input_ids"], dtype=torch.long, device=device)[None, ...]
    last_y = None
    gen = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_p=0.85, stream=True,
                         return_audio_codes=False, open_thinking=False)
    for y, _af in gen:
        if y is not None and y.numel() > 0:
            last_y = y
    if last_y is None:
        return "", 0
    answer = tokenizer.decode(last_y[0].tolist(), skip_special_tokens=True).strip()
    return answer, int(last_y.shape[1])


def load_prompts(path: str = "", limit: int = 0):
    """读取 prompt 列表：
      - 不传 path：用内置 50 问；
      - .jsonl：每行 {"prompt": "..."} 或 {"content": "..."}；
      - .txt：每行一条 prompt；
      - .json：一个字符串列表。
    """
    if not path:
        items = BUILTIN_PROMPTS
    elif path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                items.append(o.get("prompt") or o.get("content") or o.get("conversations", [{}])[0].get("content", ""))
    elif path.endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("prompts", [])
    else:
        items = [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]
    items = [p for p in items if p]
    return items[:limit] if limit and limit > 0 else items


def make_sft_record(prompt: str, answer: str):
    """构造与 minimind SFTDataset 一致的 conversations 结构。"""
    return {"conversations": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]}
#（注：内容由AI生成）
