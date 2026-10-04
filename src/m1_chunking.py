from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    from sentence_transformers import SentenceTransformer
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}

    # Tách văn bản thành các câu
    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+|\n\n', text)
        if s.strip()
    ]

    # Không có câu nào
    if not sentences:
        return []

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Chuyển từng câu thành vector
    embeddings = model.encode(sentences)

    # Bắt đầu group đầu tiên
    groups = [[sentences[0]]]

    for i in range(1, len(sentences)):
        previous_embedding = embeddings[i - 1]
        current_embedding = embeddings[i]

        similarity = dot(previous_embedding, current_embedding) / (
            norm(previous_embedding) * norm(current_embedding) + 1e-9
        )

        if similarity < threshold:
            # Chủ đề thay đổi → tạo chunk mới
            groups.append([sentences[i]])
        else:
            # Vẫn cùng chủ đề → thêm vào chunk hiện tại
            groups[-1].append(sentences[i])

    # Chuyển các group thành Chunk
    chunks = []

    for index, group in enumerate(groups):
        chunks.append(
            Chunk(
                text=" ".join(group),
                metadata={
                    **metadata,
                    "strategy": "semantic",
                    "chunk_index": index,
                },
            )
        )

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    metadata = metadata or {}

    # Tách văn bản thành các paragraph
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    parents = []
    children = []

    # Bước 1: Gom paragraph thành các parent
    parent_texts = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > parent_size:
            parent_texts.append(current.strip())
            current = ""

        current += para + "\n\n"

    if current.strip():
        parent_texts.append(current.strip())


    # Bước 2: Tạo Parent Chunk
    for parent_index, parent_text in enumerate(parent_texts):
        pid = f"parent_{parent_index}"

        parent = Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "chunk_type": "parent",
                "parent_id": pid,
            },
        )

        parents.append(parent)

        # Bước 3: Chia parent thành các child nhỏ hơn
        child_paragraphs = [
            p.strip()
            for p in parent_text.split("\n\n")
            if p.strip()
        ]

        current_child = ""

        for para in child_paragraphs:
            if current_child and len(current_child) + len(para) + 2 > child_size:
                children.append(
                    Chunk(
                        text=current_child.strip(),
                        metadata={
                            **metadata,
                            "chunk_type": "child",
                        },
                        parent_id=pid,
                    )
                )
                current_child = ""

            # Nếu một paragraph tự nó lớn hơn child_size
            if len(para) > child_size:
                if current_child:
                    children.append(
                        Chunk(
                            text=current_child.strip(),
                            metadata={
                                **metadata,
                                "chunk_type": "child",
                            },
                            parent_id=pid,
                        )
                    )
                    current_child = ""

                for start in range(0, len(para), child_size):
                    piece = para[start:start + child_size].strip()

                    if piece:
                        children.append(
                            Chunk(
                                text=piece,
                                metadata={
                                    **metadata,
                                    "chunk_type": "child",
                                },
                                parent_id=pid,
                            )
                        )
            else:
                current_child += para + "\n\n"

        # Đừng quên child cuối cùng
        if current_child.strip():
            children.append(
                Chunk(
                    text=current_child.strip(),
                    metadata={
                        **metadata,
                        "chunk_type": "child",
                    },
                    parent_id=pid,
                )
            )

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    metadata = metadata or {}

    sections = re.split(
        r'(^#{1,3}\s+.+$)',
        text,
        flags=re.MULTILINE
    )

    chunks = []
    current_header = ""
    current_content = []

    for part in sections:
        if not part.strip():
            continue

        if re.match(r'^#{1,3}\s+', part):
            if current_header or current_content:
                content = "\n\n".join(current_content).strip()
                full_text = f"{current_header}\n\n{content}".strip()

                if full_text:
                    chunks.append(
                        Chunk(
                            text=full_text,
                            metadata={
                                **metadata,
                                "section": current_header.lstrip("#").strip(),
                                "strategy": "structure",
                            },
                        )
                    )

            current_header = part.strip()
            current_content = []

        else:
            current_content.append(part.strip())

    # Lưu section cuối
    if current_header or current_content:
        content = "\n\n".join(current_content).strip()
        full_text = f"{current_header}\n\n{content}".strip()

        if full_text:
            chunks.append(
                Chunk(
                    text=full_text,
                    metadata={
                        **metadata,
                        "section": current_header.lstrip("#").strip(),
                        "strategy": "structure",
                    },
                )
            )

    return chunks

# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
