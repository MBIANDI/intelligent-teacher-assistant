import json
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory


def student_dir(sid: str, user_data_dir: str) -> Path:
    d = user_data_dir / sid
    d.mkdir(exist_ok=True, parents=True)
    return d


def load_profile(sid: str, user_data_dir: str) -> dict:
    p = student_dir(sid, user_data_dir) / "profile.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "niveau": None,
        "objectifs": [],
        "preferences": [],
        "difficultes": [],
        "faits": {},
    }


def save_profile(sid: str, profile: dict, user_data_dir: str):
    p = student_dir(sid=sid, user_data_dir=user_data_dir) / "profile.json"
    p.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def load_summary(sid: str, user_data_dir: str) -> str:
    s = student_dir(sid, user_data_dir) / "memory_summary.txt"
    if s.exists():
        return s.read_text(encoding="utf-8")
    return ""


def save_summary(sid: str, summary: str, user_data_dir: str):
    s = student_dir(sid, user_data_dir) / "memory_summary.txt"
    s.write_text(summary, encoding="utf-8")


def build_memory(
    sid: str, llm: ChatOpenAI, user_data_dir: str
) -> ConversationSummaryBufferMemory:
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1200,
        return_messages=True,
    )
    existing_summary = load_summary(sid, user_data_dir)
    if existing_summary:
        memory.moving_summary_buffer = existing_summary
    return memory
