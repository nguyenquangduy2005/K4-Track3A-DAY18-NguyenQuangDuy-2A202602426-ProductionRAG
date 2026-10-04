from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed:
Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import sys
import re
import json
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tóm tắt đoạn văn sau trong 2-3 câu "
                            "ngắn gọn bằng tiếng Việt."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                max_tokens=150,
            )

            return resp.choices[0].message.content.strip()

        except Exception as e:
            print(
                f"  ⚠️  OpenAI summarize failed: {e}"
            )

    # Extractive fallback
    sentences = [
        s.strip()
        for s in text.replace("\n", " ").split(". ")
        if s.strip()
    ]

    if not sentences:
        return text

    summary = ". ".join(sentences[:2])

    if summary and not summary.endswith("."):
        summary += "."

    return summary


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(
    text: str,
    n_questions: int = 3,
) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    """

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} "
                            "câu hỏi mà đoạn văn có thể trả lời. "
                            "Trả về mỗi câu hỏi trên 1 dòng."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                max_tokens=200,
            )

            content = (
                resp.choices[0]
                .message.content
                .strip()
            )

            questions = content.split("\n")

            cleaned = []

            for q in questions:
                q = q.strip()

                if not q:
                    continue

                q = q.lstrip(
                    "0123456789.-) "
                ).strip()

                if q:
                    cleaned.append(q)

            return cleaned[:n_questions]

        except Exception as e:
            print(
                f"  ⚠️  OpenAI HyQA failed: {e}"
            )

    # Extractive fallback
    sentences = [
        s.strip()
        for s in re.split(r"[.!?\n]", text)
        if len(s.strip()) > 10
    ]

    questions = []

    for sentence in sentences[:n_questions]:
        sentence = sentence.rstrip(".?! ")

        questions.append(
            f"{sentence}?"
        )

    return questions


# ─── Technique 3: Contextual Prepend ─────────────────────


def contextual_prepend(
    text: str,
    document_title: str = "",
) -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    """

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết 1 câu ngắn mô tả đoạn văn "
                            "này nằm ở đâu trong tài liệu và "
                            "nói về chủ đề gì. "
                            "Chỉ trả về 1 câu."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Tài liệu: {document_title}\n\n"
                            f"Đoạn văn:\n{text}"
                        ),
                    },
                ],
                max_tokens=80,
            )

            context = (
                resp.choices[0]
                .message.content
                .strip()
            )

            return f"{context}\n\n{text}"

        except Exception as e:
            print(
                f"  ⚠️  OpenAI contextual failed: {e}"
            )

    # Simple fallback
    if document_title:
        prefix = f"Trích từ {document_title}. "
        return f"{prefix}{text}"

    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động:
    topic, entities, category, language.
    """

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Trích xuất metadata từ đoạn văn. "
                            "Chỉ trả về JSON hợp lệ theo format: "
                            '{"topic": "...", '
                            '"entities": ["..."], '
                            '"category": '
                            '"policy|hr|it|finance", '
                            '"language": "vi|en"}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                max_tokens=150,
                response_format={
                    "type": "json_object"
                },
            )

            content = (
                resp.choices[0]
                .message.content
            )

            return json.loads(content)

        except Exception as e:
            print(
                f"  ⚠️  OpenAI metadata failed: {e}"
            )

    # Fallback metadata
    return {
        "topic": "general",
        "entities": [],
        "category": "policy",
        "language": "vi",
    }


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(
    text: str,
    source: str,
) -> dict:
    """
    Single LLM call:
    summary + questions + context + metadata.

    Cost optimization:
    1 API call thay vì 4 calls riêng lẻ.
    """

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)

            system_prompt = """
Phân tích đoạn văn và chỉ trả về JSON hợp lệ theo cấu trúc:

{
  "summary": "tóm tắt 2-3 câu",
  "questions": [
    "câu hỏi 1",
    "câu hỏi 2",
    "câu hỏi 3"
  ],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {
    "topic": "...",
    "entities": ["..."],
    "category": "policy|hr|it|finance",
    "language": "vi|en"
  }
}
""".strip()

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Tài liệu: {source}\n\n"
                            f"Đoạn văn:\n{text}"
                        ),
                    },
                ],
                max_tokens=400,
                response_format={
                    "type": "json_object"
                },
            )

            content = (
                resp.choices[0]
                .message.content
            )

            result = json.loads(content)

            # Đảm bảo đủ 4 fields
            return {
                "summary": result.get(
                    "summary",
                    "",
                ),
                "questions": result.get(
                    "questions",
                    [],
                ),
                "context": result.get(
                    "context",
                    "",
                ),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
            }

        except Exception as e:
            print(
                f"  ⚠️  Enrichment API failed: {e}"
            )

    # ─── Fallback không cần API ───

    summary = summarize_chunk(text)

    questions = generate_hypothesis_questions(
        text,
        n_questions=3,
    )

    if source:
        context = (
            f"Đoạn văn này được trích từ {source}."
        )
    else:
        context = (
            "Đoạn văn này cung cấp thông tin "
            "liên quan đến tài liệu hiện tại."
        )

    metadata = extract_metadata(text)

    return {
        "summary": summary,
        "questions": questions,
        "context": context,
        "metadata": metadata,
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Có 2 chế độ:
    - methods cụ thể:
      ["summary"], ["contextual"], ...
    - methods=["combined"] hoặc None:
      1 API call duy nhất cho mỗi chunk.
    """

    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []

    for i, chunk in enumerate(chunks):

        text = chunk["text"]

        source = (
            chunk
            .get("metadata", {})
            .get("source", "")
        )

        if use_combined:

            result = _enrich_single_call(
                text,
                source,
            )

            summary = result.get(
                "summary",
                "",
            )

            questions = result.get(
                "questions",
                [],
            )

            context_line = result.get(
                "context",
                "",
            )

            if context_line:
                enriched_text = (
                    f"{context_line}\n\n{text}"
                )
            else:
                enriched_text = text

            auto_meta = result.get(
                "metadata",
                {},
            )

        else:

            summary = (
                summarize_chunk(text)
                if "summary" in methods
                else ""
            )

            questions = (
                generate_hypothesis_questions(text)
                if "hyqa" in methods
                else []
            )

            enriched_text = (
                contextual_prepend(
                    text,
                    source,
                )
                if "contextual" in methods
                else text
            )

            auto_meta = (
                extract_metadata(text)
                if "metadata" in methods
                else {}
            )

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={
                    **chunk.get(
                        "metadata",
                        {},
                    ),
                    **auto_meta,
                },
                method="+".join(methods),
            )
        )

        if (
            (i + 1) % 10 == 0
            or (i + 1) == len(chunks)
        ):
            print(
                f"  Enriched "
                f"{i + 1}/{len(chunks)} "
                f"chunks...",
                flush=True,
            )

    return enriched


# ─── Main ────────────────────────────────────────────────


if __name__ == "__main__":

    sample = (
        "Nhân viên chính thức được nghỉ phép năm "
        "12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ phép tăng thêm 1 ngày "
        "cho mỗi 5 năm thâm niên công tác."
    )

    print(
        "=== Enrichment Pipeline Demo ===\n"
    )

    print(
        f"Original: {sample}\n"
    )

    s = summarize_chunk(sample)

    print(
        f"Summary: {s}\n"
    )

    qs = generate_hypothesis_questions(
        sample
    )

    print(
        f"HyQA questions: {qs}\n"
    )

    ctx = contextual_prepend(
        sample,
        "Sổ tay nhân viên VinUni 2024",
    )

    print(
        f"Contextual: {ctx}\n"
    )

    meta = extract_metadata(sample)

    print(
        f"Auto metadata: {meta}"
    )