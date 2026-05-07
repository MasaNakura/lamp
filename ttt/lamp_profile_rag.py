"""LaMP-aligned profile retrieval for TTT (same corpus/query split as ``prompts/prompts.py``)."""
from __future__ import annotations

import random
import sys
from typing import Any

import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer


def _batchify(lst: list[str], batch_size: int) -> list[list[str]]:
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def _mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


def _extract_after_paper(input_string: str) -> str | None:
    """Match LaMP ``extract_after_paper`` (None if marker missing)."""
    article_index = input_string.find("paper:")
    if article_index == -1:
        return None
    return input_string[article_index + len("paper:") :].strip()


def _extract_after_colon(input_string: str) -> str | None:
    """Match LaMP ``extract_after_colon`` (None if marker missing)."""
    article_index = input_string.find(":")
    if article_index == -1:
        return None
    return input_string[article_index + 1 :].strip()


def _corpus_and_query(task: str, inp: str, profile: list[dict[str, Any]]) -> tuple[list[str], str]:
    if task == "LaMP-5":
        corpus = [f'{p.get("title", "")} {p.get("abstract", "")}' for p in profile]
        query = _extract_after_paper(inp)
    elif task == "LaMP-7":
        corpus = [f'{p.get("text", "")}' for p in profile]
        query = _extract_after_colon(inp)
    else:
        raise ValueError(task)
    if query is None or not str(query).strip():
        query = inp.strip()
    return corpus, query


class LampProfileRAG:
    """
    Select a subset of profile dicts for TTT / meta-training streams, mirroring LaMP RAG
    (BM25 / Contriever / random / recency) with ``ranked=False``.
    """

    def __init__(
        self,
        task: str,
        *,
        num_retrieved: int = 3,
        retriever: str = "bm25",
        ranked: bool = False,
        device: torch.device | None = None,
        cache_dir: str | None = None,
    ):
        self.task = task
        self.num_retrieved = max(1, int(num_retrieved))
        self.retriever = retriever
        self.ranked = ranked
        if device is None:
            self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.cache_dir = cache_dir
        self._ct_tok: AutoTokenizer | None = None
        self._ct_model: torch.nn.Module | None = None
        if retriever == "contriever" and not ranked:
            self._ct_tok = AutoTokenizer.from_pretrained("facebook/contriever", cache_dir=cache_dir)
            self._ct_model = AutoModel.from_pretrained("facebook/contriever", cache_dir=cache_dir)
            self._ct_model.to(self.device).eval()
        if ranked:
            print(
                "[LampProfileRAG] ranked=True is not replicated for TTT streams; using unranked retrieval.",
                file=sys.stderr,
            )

    def select(self, inp: str, profile: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not profile:
            return []
        k = min(self.num_retrieved, len(profile))

        corpus, query = _corpus_and_query(self.task, inp, profile)

        if self.retriever == "random":
            return random.sample(profile, k=k) if k < len(profile) else list(profile)

        if self.retriever == "recency":
            if self.task == "LaMP-5" and profile and all("date" in p for p in profile):
                sorted_p = sorted(profile, key=lambda x: tuple(map(int, str(x["date"]).split("-"))))
                return sorted_p[-k:][::-1]
            return profile[-k:][::-1]

        if self.retriever == "bm25":
            tokenized_corpus = [doc.split() for doc in corpus]
            if not any(tokenized_corpus):
                return list(profile[:k])
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split()
            if not tokenized_query:
                tokenized_query = query.replace("\n", " ").strip() or ["."]
            return bm25.get_top_n(tokenized_query, profile, n=k)

        if self.retriever == "contriever":
            if self.ranked or self._ct_model is None or self._ct_tok is None:
                tokenized_corpus = [doc.split() for doc in corpus]
                if not any(tokenized_corpus):
                    return list(profile[:k])
                bm25 = BM25Okapi(tokenized_corpus)
                tq = query.split() or query.replace("\n", " ").strip() or ["."]
                return bm25.get_top_n(tq, profile, n=k)
            return self._contriever_top_k(corpus, profile, query, k)

        raise ValueError(f"Unknown retriever: {self.retriever!r}")

    @torch.inference_mode()
    def _contriever_top_k(
        self,
        corpus: list[str],
        profile: list[dict[str, Any]],
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        assert self._ct_model is not None and self._ct_tok is not None
        dev = self.device
        ct = self._ct_tok
        model = self._ct_model
        query_tokens = ct([query], padding=True, truncation=True, return_tensors="pt").to(dev)
        out_q = model(**query_tokens)
        q_emb = _mean_pooling(out_q.last_hidden_state, query_tokens["attention_mask"])
        scores: list[float] = []
        for batch in _batchify(corpus, 4):
            if not batch:
                continue
            tokens_batch = ct(batch, padding=True, truncation=True, return_tensors="pt").to(dev)
            out_b = model(**tokens_batch)
            emb_b = _mean_pooling(out_b.last_hidden_state, tokens_batch["attention_mask"])
            temp_scores = q_emb.squeeze() @ emb_b.T
            scores.extend(temp_scores.tolist())
        if not scores:
            return list(profile[:k])
        nk = min(k, len(scores))
        topk_values, topk_indices = torch.topk(torch.tensor(scores, device=dev), nk)
        _ = topk_values
        return [profile[int(m)] for m in topk_indices.tolist()]
