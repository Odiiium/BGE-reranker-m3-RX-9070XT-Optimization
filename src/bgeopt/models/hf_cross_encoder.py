"""V0 baseline: HuggingFace transformers cross-encoder (XLMRobertaForSequenceClassification).

Optimized variants subclass this and override `build_model()` (e.g. torch.compile, patched modules),
`forward_batch()` (custom kernels) or `prepare()` / `predict()` (e.g. unpadded varlen inputs).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bgeopt.models.base import BaseReranker, PreparedInput, RerankerConfig
from bgeopt.models.registry import register_reranker

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class HFCrossEncoderConfig(RerankerConfig):
    model_id: str = "BAAI/bge-reranker-v2-m3"
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "float16"
    attn_implementation: str = "sdpa"
    max_length: int = 512
    query_max_length: int = 384
    batch_size: int = 32
    sort_by_length: bool = True


@dataclass
class PaddedBatches:
    batches: list[dict[str, torch.Tensor]]  # CPU tensors, one dict per mini-batch
    order: np.ndarray  # order[i] = original index of the i-th pair in batch order


@register_reranker("hf-cross-encoder")
class HFCrossEncoderReranker(BaseReranker):
    config_class = HFCrossEncoderConfig
    config: HFCrossEncoderConfig

    def __init__(self, config: HFCrossEncoderConfig):
        super().__init__(config)
        if config.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm device not visible to PyTorch; under WSL2 export HSA_ENABLE_DXG_DETECTION=1")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
        self._prefix, self._middle, self._suffix = self._pair_template()
        self.model = self.build_model()

    def _pair_template(self) -> tuple[list[int], list[int], list[int]]:
        """Special tokens around a (query, passage) pair, e.g. XLM-R: <s> q </s></s> p </s>.
        transformers 5 dropped prepare_for_model / build_inputs_with_special_tokens, so derive it from an encoding."""
        first, second = "alpha beta", "gamma delta"
        a = self.tokenizer(first, add_special_tokens=False)["input_ids"]
        b = self.tokenizer(second, add_special_tokens=False)["input_ids"]
        pair = self.tokenizer(first, second)["input_ids"]
        a_start = next(i for i in range(len(pair)) if pair[i:i + len(a)] == a)
        b_start = next(i for i in range(a_start + len(a), len(pair)) if pair[i:i + len(b)] == b)
        template = pair[:a_start], pair[a_start + len(a):b_start], pair[b_start + len(b):]
        if template[0] + a + template[1] + b + template[2] != pair:
            raise RuntimeError("could not derive the pair template from the tokenizer")
        return template

    def build_model(self) -> torch.nn.Module:
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            dtype=DTYPES[self.config.dtype],
            attn_implementation=self.config.attn_implementation,
        )
        return model.to(self.device).eval()

    # ------------------------------------------------------------------------------------------------------
    # CPU stage — same input construction as FlagEmbedding's FlagReranker.compute_score
    # ------------------------------------------------------------------------------------------------------

    def prepare(self, query: str, documents: Sequence[str]) -> PreparedInput:
        cfg = self.config
        query_ids = self.tokenizer(query, add_special_tokens=False, truncation=True,
                                   max_length=cfg.query_max_length)["input_ids"]
        doc_ids = self.tokenizer(list(documents), add_special_tokens=False, truncation=True,
                                 max_length=cfg.max_length)["input_ids"]
        # truncation="only_second": the passage is cut from the right so the whole pair fits max_length
        head = self._prefix + query_ids + self._middle
        passage_budget = max(cfg.max_length - len(head) - len(self._suffix), 0)
        pairs = [head + ids[:passage_budget] + self._suffix for ids in doc_ids]

        lengths = np.fromiter((len(p) for p in pairs), dtype=np.int64, count=len(pairs))
        order = np.argsort(-lengths, kind="stable") if cfg.sort_by_length else np.arange(len(pairs))

        pad_id = self.tokenizer.pad_token_id
        batches, padded_tokens = [], 0
        for start in range(0, len(order), cfg.batch_size):
            chunk = order[start:start + cfg.batch_size]
            width = int(lengths[chunk].max())
            input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, index in enumerate(chunk):  # right padding, as the tokenizer does
                input_ids[row, :lengths[index]] = torch.tensor(pairs[index])
                attention_mask[row, :lengths[index]] = 1
            padded_tokens += input_ids.numel()
            batches.append({"input_ids": input_ids, "attention_mask": attention_mask})

        return PreparedInput(
            payload=PaddedBatches(batches=batches, order=order),
            num_pairs=len(pairs),
            num_tokens=int(lengths.sum()),
            num_padded_tokens=padded_tokens,
        )

    # ------------------------------------------------------------------------------------------------------
    # device stage
    # ------------------------------------------------------------------------------------------------------

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Logits [batch] for one padded mini-batch already on the device."""
        return self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.view(-1)

    @torch.inference_mode()
    def predict(self, prepared: PreparedInput) -> np.ndarray:
        payload: PaddedBatches = prepared.payload
        logits = [
            self.forward_batch({k: v.to(self.device, non_blocking=True) for k, v in batch.items()}).float()
            for batch in payload.batches
        ]
        in_batch_order = torch.cat(logits).cpu().numpy()  # .cpu() synchronizes the device
        scores = np.empty_like(in_batch_order)
        scores[payload.order] = in_batch_order
        return scores

    # ------------------------------------------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------------------------------------------

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_mb(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return torch.cuda.max_memory_allocated(self.device) / 2**20

    def weights_sha256(self) -> str | None:
        """LFS blobs in the HF cache are named by their sha256, so no hashing of the 2.3 GB file is needed."""
        path = try_to_load_from_cache(self.config.model_id, "model.safetensors", revision=self.config.revision)
        return os.path.basename(os.path.realpath(path)) if isinstance(path, str) else None

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            weights_sha256=self.weights_sha256(),
            num_parameters=sum(p.numel() for p in self.model.parameters()),
            model_dtype=str(next(self.model.parameters()).dtype),
            attn_implementation=self.model.config._attn_implementation,
            transformers_version=transformers.__version__,
        )
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            info.update(device_name=props.name, device_arch=getattr(props, "gcnArchName", None),
                        device_memory_mb=props.total_memory // 2**20)
        return info
