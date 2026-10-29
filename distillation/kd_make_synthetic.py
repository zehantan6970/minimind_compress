"""
kd_make_synthetic.py —— 方案 C（混合标签）：确定性模板标签生成器

为什么需要它：
  MoE 教师（minimind-3o-moe）是通用闲聊模型，对"开灯/关灯/调亮/调暗"这类
  台灯指令只会打太极（"It seems like you're referring to..."），拿它当标签会
  教坏学生。而产品上这些指令本来就该映射到固定动作词表，回答必须确定、永远正确。
  所以：
    - 指令类（开关/亮度/色温/点头/停止/语音控制）→ 模板化短确认，本脚本生成；
    - 越界类（天气/时间/新闻/放音乐/闹钟）       → 固定礼貌拒绝模板，本脚本生成；
    - 身份/能力/致谢等高频产品话术               → 固定模板，本脚本生成；
    - 闲聊类（笑话/情绪/鼓励）                   → 继续用 MoE 教师，先用
        kd_teacher_generate.py 生成 kd_chat.jsonl，
        再用 --chat_file 合并进同一份训练集。
  合并后格式与教师生成的 kd_sft*.jsonl 完全一致（conversations），
  kd_train_student.py / kd_eval*.py 不用改。

直接生成（只有模板部分）：
  python kd_make_synthetic.py --out kd_sft_lamp.jsonl
  每类意图给了大量同义改写（prompt），目的是让学生学会"按意图路由"，而不是背句子。
生成模板 + 合并教师闲聊：
  python kd_make_synthetic.py --out kd_sft_lamp.jsonl --chat_file kd_chat.jsonl
  （先让教师只对"闲聊 prompt"生成 kd_chat.jsonl，别把指令 prompt 喂给教师）
"""
import os
import json
import random
import argparse

from kd_common import make_sft_record  # 复用统一的 conversations 行格式，保证下游不用改

# ---------------------------------------------------------------------------
# 意图表：每个意图 = 多种用户说法(prompts) × 多个自然短确认(replies)
# prompts 覆盖同一意图的大量同义改写（关键：数量要够，学生才能对没见过的说法泛化）；
# replies 给 3~5 个自然说法，训练时随机选一个，避免学生只会机械复读单句。
# 这是"领域专家规则"（Plan B / Guardian 的 ACT），不是 LLM 自由发挥。
# ---------------------------------------------------------------------------
INTENTS = {
    # 1) 开灯
    "turn_on": {
        "prompts": [
            "Turn on the light.", "Please turn on the light.", "Can you turn on the light?",
            "Could you turn on the light?", "Switch on the light.", "Please switch on the light.",
            "Can you switch on the light?", "Turn the light on.", "Light it up.",
            "Hey lamp, turn on the light.",
            # —— 更多同义改写（扩充泛化） ——
            "I need some light.", "Can you give me light?", "Light it up, please.",
            "Please put the light on.", "Could you get the light on?", "Turn on the desk lamp.",
            "I'd like the light on.", "Light up the room.", "Let's have the light on.",
            "Can you light things up?",
        ],
        "replies": [
            "Sure, I'll turn on the light.", "Okay, turning the light on.",
            "Done, the light is on.", "On it.",
        ],
    },
    # 2) 关灯
    "turn_off": {
        "prompts": [
            "Turn off the light.", "Please turn off the light.", "Can you turn off the light?",
            "Could you turn off the light?", "Switch off the light.", "Please switch off the light.",
            "Turn the light off.", "Shut the light off.", "Lights off.",
            "Hey lamp, turn off the light.",
            # —— 更多同义改写 ——
            "Can you shut the lights?", "Please turn it off.", "Turn the desk lamp off.",
            "Switch it off now.", "I don't need the light anymore.", "Could you kill the lamp?",
            "Put the light out.", "Turn everything off.", "Lights out now.",
            "Turn the light off for me.",
        ],
        "replies": [
            "Sure, I'll turn off the light.", "Okay, turning the light off.",
            "Done, the light is off.", "Off it goes.",
        ],
    },
    # 3) 调亮
    "brighter": {
        "prompts": [
            "Make it brighter.", "Make the lamp brighter.", "Brighter, please.",
            "Turn up the brightness.", "More light, please.", "It's too dark here.",
            "Can you make it brighter?", "Make the light brighter.",
            # —— 更多同义改写 ——
            "I can't see well here.", "Make this brighter.", "Turn it up.",
            "More light on my book.", "Crank up the brightness.", "It's too dim in here.",
            "I need brighter light.", "Boost the light.",
        ],
        "replies": [
            "Sure, making it brighter.", "Okay, increasing the brightness.",
            "Done, brighter now.", "Making it brighter.",
        ],
    },
    # 4) 调暗
    "dimmer": {
        "prompts": [
            "Dim the light.", "Make it dimmer.", "Dim it, please.",
            "Turn down the brightness.", "Less light, please.", "It's too bright.",
            "Can you make the light softer?", "Make the light softer.",
            # —— 更多同义改写 ——
            "Turn it down.", "It's glaring.", "Dim the desk lamp.",
            "Less brightness, please.", "Too much light.", "Ease off the brightness.",
            "Turn the light way down.", "Make it softer on the eyes.",
        ],
        "replies": [
            "Sure, dimming it.", "Okay, lowering the brightness.",
            "Done, softer now.", "Dimming it.",
        ],
    },
    # 5) 色温（暖光 / 阅读模式）
    "scene_warm": {
        "prompts": [
            "Switch to warm light.", "Make it warm.", "Set a warm white.",
            "Switch to reading mode.", "I want a warmer color.", "Use warm light.",
            # —— 更多同义改写 ——
            "Give me a warm light.", "Set it to warm.", "I want a cozy glow.",
            "Use a warmer color.", "Switch to a soft warm light.", "Make it warm and cozy.",
            "Warm tone, please.", "Set the warm white mode.",
        ],
        "replies": [
            "Sure, switching to warm light.", "Okay, warm white now.",
            "Done, here's a warmer tone.", "Switching to warm light.",
        ],
    },
    # 6) 点头（物理确认，Guardian 靠它表示"我听到了"）
    "nod": {
        "prompts": [
            "Nod the lamp.", "Nod once.", "Give me a nod.", "Can you nod?",
            "Show me you heard me.",
            # —— 更多同义改写 ——
            "Nod your head.", "Nod once for me.", "Can you nod your head?",
            "Give me a little nod.", "Nod if you can hear me.", "Show me a nod.",
        ],
        "replies": [
            "Sure, nodding once.", "Okay, here's a nod.", "Nodding.",
        ],
    },
    # 7) 停止
    "stop": {
        "prompts": [
            "Stop.", "Stop that.", "Stop it.", "Enough.", "Be quiet.", "Stop talking.",
            # —— 更多同义改写 ——
            "Stop doing that.", "That's enough.", "Please stop.", "Stop now.",
            "Enough, please.", "Quiet, please.", "That will do.", "Stop speaking.",
        ],
        "replies": [
            "Okay, stopped.", "Sure, I'll stop.", "No problem, stopping.",
        ],
    },
    # 8) 语音控制：重复 / 说慢 / 大声
    "repeat": {
        "prompts": [
            "Please repeat that.", "Can you repeat it?", "Say it again, please.",
            "I didn't hear you.", "Pardon?", "I can't hear you.",
            # —— 更多同义改写 ——
            "I missed that.", "What was that again?", "Repeat it, please.",
            "Can you say it again?", "One more time, please.", "Say it once more.",
            "Repeat what you said.", "Run that by me again.",
        ],
        "replies": [
            "No problem, I'll say it again.", "Sure, repeating that.",
            "Okay, here it is again.",
        ],
    },
    "speak_slower": {
        "prompts": [
            "Speak slower, please.", "Slow down, please.", "Can you talk more slowly?",
            "You're speaking too fast.",
            # —— 更多同义改写 ——
            "Slow down a bit.", "Slow down your speech.", "Speak more slowly.",
            "Say it slower.", "Not so fast, please.", "Take it slower.",
        ],
        "replies": [
            "Sure, I'll speak slower.", "Okay, talking more slowly.", "Got it, slowing down.",
        ],
    },
    "speak_louder": {
        "prompts": [
            "Speak louder, please.", "Talk louder, please.", "Can you speak up?",
            "Turn up your volume.", "It's too quiet.",
            # —— 更多同义改写 ——
            "Speak up, please.", "Can't hear you.", "A little louder?",
            "Talk more loudly.", "Turn up your voice.", "More volume, please.",
        ],
        "replies": [
            "Sure, speaking louder.", "Okay, turning up the volume.", "Got it, louder now.",
        ],
    },
    # 9) 越界能力：固定礼貌拒绝（说明我是台灯、能做什么）
    "refuse": {
        "prompts": [
            "What's the weather like?", "Will it rain tomorrow?", "What's the temperature?",
            "What time is it?", "What day is it today?", "Tell me the news.",
            "Play some music.", "Sing a song for me.", "Set an alarm for 7 am.",
            "What is 2 plus 3?", "Call my friend.", "Order me a pizza.",
            # —— 更多同义改写 ——
            "What's the weather tomorrow?", "Do I need an umbrella?", "What time does it get dark?",
            "Play a song.", "Play some music for me.", "Set an alarm.",
            "Remind me at six.", "Send a message for me.", "What is 10 times 10?",
            "Can you order something?",
        ],
        "replies": [
            "I'm just a smart lamp, so I can't do that. I can turn the light on or off, "
            "adjust brightness and color, nod to show I heard you, and chat with you.",
            "Sorry, I can't do that — I'm a lamp. I can control the light or talk with you.",
            "That's beyond what I can do. I can switch the light, change brightness and "
            "color, and keep you company.",
        ],
    },
    # 10) 身份 / 能力（产品话术必须确定）
    "identity": {
        "prompts": [
            "What is the smart lamp for?", "What can you do?", "What are you?",
            "Introduce yourself.", "What do you do?",
            # —— 更多同义改写 ——
            "What are you exactly?", "Are you a lamp?", "What are your functions?",
            "Explain what you do.", "What can you help with?", "Do you control the light?",
        ],
        "replies": [
            "I'm a smart lamp. I can turn the light on or off, adjust brightness and color, "
            "nod to show I heard you, and chat with you.",
            "I'm your smart desk lamp. I control the light — on/off, brightness, warm color — "
            "and I can talk with you.",
        ],
    },
    # 11) 问候 / 致谢 / 道别（短而确定）
    "greeting": {
        "prompts": [
            "Hello.", "Hi.", "Hey.", "Good morning.", "Good evening.",
            # —— 更多同义改写 ——
            "Hello there.", "Hi there.", "Good day.", "Hey lamp.", "Hi lamp.", "Howdy.",
        ],
        "replies": [
            "Hi! I'm your smart lamp. Ask me to turn the light on or off, or just chat.",
            "Hello there!", "Hi, nice to see you.",
        ],
    },
    "thanks": {
        "prompts": [
            "Thank you.", "Thanks.", "Thank you very much.", "Thanks a lot.",
            # —— 更多同义改写 ——
            "Thanks, that helps.", "I appreciate it.", "Much appreciated.",
            "Thank you so much.", "That's helpful, thanks.", "You've been helpful.",
        ],
        "replies": ["You're welcome!", "Happy to help.", "Anytime."],
    },
    "goodbye": {
        "prompts": [
            "Goodbye.", "Bye.", "See you later.", "That's all, thanks.",
            # —— 更多同义改写 ——
            "Bye for now.", "See you tomorrow.", "Catch you later.",
            "I'm heading out.", "That's all for now.", "Farewell.", "Until next time.",
        ],
        "replies": ["Goodbye!", "See you later!", "Bye for now."],
    },
}


def build_records(repeat=1, rng=random):
    """按意图表生成模板记录：每条 prompt 出现 repeat 次，每次随机选一个回复。

    返回 (records, counts)：records 是 conversations 行列表，counts 是每意图条数。
    """
    records, counts = [], {}
    for intent, spec in INTENTS.items():
        n = 0
        for _ in range(repeat):                       # repeat>1 时同一 prompt 多次出现，
            for prompt in spec["prompts"]:           # 每次随机选不同回复，增加说法多样性
                reply = rng.choice(spec["replies"])  # 例：turn_on 的 "Switch on the light." → "Okay, turning the light on."
                records.append(make_sft_record(prompt, reply))
                n += 1
        counts[intent] = n
    return records, counts


def load_chat_records(path):
    """读教师生成的闲聊文件（已是 conversations 行，每行一个 JSON），原样返回。"""
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dedup_by_prompt(records):
    """按 user 文本去重，保留第一次出现；返回去重后的列表。"""
    seen, kept = set(), []
    for r in records:
        key = r["conversations"][0]["content"]
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kd_sft_lamp.jsonl", help="合并后的训练集输出")
    ap.add_argument("--chat_file", default="", help="教师生成的闲聊 SFT（kd_chat.jsonl），可选")
    ap.add_argument("--repeat", type=int, default=1, help="每条 prompt 重复几次（不同随机回复）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    ap.add_argument("--no_dedup", action="store_true", help="关闭按 prompt 去重")
    ap.add_argument("--no_shuffle", action="store_true", help="关闭随机打乱")
    args = ap.parse_args()

    rng = random.Random(args.seed)                   # 独立随机源，结果可复现
    records, counts = build_records(args.repeat, rng)
    n_template = len(records)

    chat = load_chat_records(args.chat_file)        # 闲聊来自教师（GENERATE），模板来自规则（ACT）
    n_chat = len(chat)
    records = records + chat

    if not args.no_dedup and args.repeat == 1:
        # repeat=1 时 prompt 本身唯一；合并闲聊后可能有重复，统一去重
        records = dedup_by_prompt(records)
    if not args.no_shuffle:
        rng.shuffle(records)                        # 打乱，避免同一意图样本连续出现

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[synthetic] 模板意图 {len(INTENTS)} 类，模板样本 {n_template} 条，"
          f"教师闲聊 {n_chat} 条，写出（去重后）{len(records)} 条 -> {args.out}")
    for k, v in counts.items():
        print(f"  - {k:14s} {v} 条")


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
