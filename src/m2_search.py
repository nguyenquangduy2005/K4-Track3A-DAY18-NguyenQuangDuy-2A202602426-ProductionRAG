from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    BM25_TOP_K,
    DENSE_TOP_K,
    HYBRID_TOP_K,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text for BM25."""
    from underthesea import word_tokenize

    segmented = word_tokenize(text, format="text")

    # underthesea có thể tạo từ ghép dạng "nghỉ_phép".
    # Đổi "_" thành space theo yêu cầu của bài.
    return segmented.replace("_", " ")


class BM25Search:

    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks

        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).lower().split()
            for chunk in chunks
        ]

        if not self.corpus_tokens:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K
    ) -> list[SearchResult]:
        """Search using BM25."""

        if self.bm25 is None:
            return []

        tokenized_query = (
            segment_vietnamese(query)
            .lower()
            .split()
        )

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for i in top_indices:

            # Bỏ document không liên quan
            if scores[i] <= 0:
                continue

            document = self.documents[i]

            results.append(
                SearchResult(
                    text=document["text"],
                    score=float(scores[i]),
                    metadata=document.get("metadata", {}),
                    method="bm25",
                )
            )

        return results


class DenseSearch:

    def __init__(self):
        from qdrant_client import QdrantClient

        try:
            self.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                timeout=2,
            )

            # Kiểm tra Qdrant server có hoạt động không
            self.client.get_collections()

        except Exception:
            # Fallback để project vẫn chạy khi chưa bật Docker/Qdrant
            self.client = QdrantClient(":memory:")

        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                EMBEDDING_MODEL
            )

        return self._encoder

    def index(
        self,
        chunks: list[dict],
        collection: str = COLLECTION_NAME
    ) -> None:
        """Index chunks into Qdrant."""

        from qdrant_client.models import (
            Distance,
            VectorParams,
            PointStruct,
        )

        if not chunks:
            return

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        # Encode toàn bộ documents
        vectors = self._get_encoder().encode(
            texts,
            show_progress_bar=True,
        )

        # BGE-M3 theo config là 1024 chiều.
        # Lấy dimension thực tế để tránh lỗi mismatch.
        vector_size = len(vectors[0])

        # Tạo lại collection
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        points = []

        for i, (chunk, vector) in enumerate(
            zip(chunks, vectors)
        ):
            payload = {
                **chunk.get("metadata", {}),
                "text": chunk["text"],
            }

            points.append(
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=collection,
            points=points,
        )

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME
    ) -> list[SearchResult]:
        """Search using dense vectors."""

        query_vector = (
            self._get_encoder()
            .encode(query)
            .tolist()
        )

        # qdrant-client mới dùng query_points()
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )

        results = []

        for point in response.points:

            payload = point.payload or {}

            results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=float(point.score),
                    metadata={
                        key: value
                        for key, value in payload.items()
                        if key != "text"
                    },
                    method="dense",
                )
            )

        return results


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K
) -> list[SearchResult]:
    """
    Merge ranked lists using Reciprocal Rank Fusion.

    score(d) = Σ 1 / (k + rank + 1)
    """

    rrf_scores = {}

    for result_list in results_list:

        for rank, result in enumerate(result_list):

            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "result": result,
                }

            rrf_scores[result.text]["score"] += (
                1.0 / (k + rank + 1)
            )

    ranked = sorted(
        rrf_scores.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    results = []

    for item in ranked[:top_k]:

        original = item["result"]

        results.append(
            SearchResult(
                text=original.text,
                score=float(item["score"]),
                metadata=original.metadata,
                method="hybrid",
            )
        )

    return results


class HybridSearch:
    """
    Combines BM25 + Dense + RRF.
    """

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:

        self.bm25.index(chunks)

        self.dense.index(chunks)

    def search(
        self,
        query: str,
        top_k: int = HYBRID_TOP_K
    ) -> list[SearchResult]:

        bm25_results = self.bm25.search(
            query,
            top_k=BM25_TOP_K,
        )

        dense_results = self.dense.search(
            query,
            top_k=DENSE_TOP_K,
        )

        return reciprocal_rank_fusion(
            [
                bm25_results,
                dense_results,
            ],
            top_k=top_k,
        )


if __name__ == "__main__":

    text = "Nhân viên được nghỉ phép năm"

    print(f"Original:  {text}")

    print(
        f"Segmented: {segment_vietnamese(text)}"
    )