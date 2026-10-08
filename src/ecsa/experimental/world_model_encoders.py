"""Frozen causal-language-model states and next-observation likelihood for ECSA.

World-model checkpoints are *generative* language models, not pretrained
symmetric embedding encoders. This experimental port extracts full contextual
hidden states and evaluates teacher-forced P(observation_after | observation_before,
action), strictly separately from the KAN pair-compatibility experiment.

No KAN gradient may modify the frozen backbone. Model downloads are opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Protocol, runtime_checkable

from ecsa.world_model.text_relations import TextRelationSample

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .token_state_relations import TokenStates


class Candidate(StrEnum):
    BEHR_TEXTWORLD = "behr_textworld"
    BASE_QWEN25 = "base_qwen25"
    SMALL_QWEN25 = "small_qwen25"
    AGENTWORLD = "agentworld"


@dataclass(frozen=True)
class ModelSpec:
    candidate: Candidate
    model_id: str
    kind: str
    approximate_parameters: str
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, Candidate):
            raise TypeError("Candidate enum required")
        if self.kind not in ("causal_hidden_states", "remote_simulator_only"):
            raise ValueError("unknown model capability")


SPECS: dict[Candidate, ModelSpec] = {
    Candidate.BEHR_TEXTWORLD: ModelSpec(
        Candidate.BEHR_TEXTWORLD,
        "Ricardo-H/BehR-WorldModel-Textworld-Qwen2.5-7B",
        "causal_hidden_states", "7B",
        "https://huggingface.co/Ricardo-H/BehR-WorldModel-Textworld-Qwen2.5-7B",
    ),
    Candidate.BASE_QWEN25: ModelSpec(
        Candidate.BASE_QWEN25, "Qwen/Qwen2.5-7B", "causal_hidden_states", "7B",
        "https://huggingface.co/Qwen/Qwen2.5-7B",
    ),
    Candidate.SMALL_QWEN25: ModelSpec(
        Candidate.SMALL_QWEN25, "Qwen/Qwen2.5-0.5B", "causal_hidden_states",
        "0.5B; non-world-model CI control",
        "https://huggingface.co/Qwen/Qwen2.5-0.5B",
    ),
    Candidate.AGENTWORLD: ModelSpec(
        Candidate.AGENTWORLD, "Qwen/Qwen-AgentWorld-35B-A3B",
        "remote_simulator_only", "35B total / 3B active",
        "https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B",
    ),
}


@runtime_checkable
class CausalTokenizer(Protocol):
    def __call__(self, text: str, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class CausalLMConfig:
    candidate: Candidate = Candidate.BEHR_TEXTWORLD
    model_id: str | None = None
    revision: str | None = None
    device: str = "cuda"
    max_length: int = 512
    quantized_4bit: bool = False
    allow_large_cpu: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, Candidate):
            raise TypeError("typed candidate required")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if type(self.max_length) is not int or self.max_length < 32:
            raise ValueError("max_length must be >=32")
        if self.model_id is not None and not self.model_id.strip():
            raise ValueError("model ID cannot be empty")
        if self.quantized_4bit and self.device != "cuda":
            raise ValueError("4-bit bitsandbytes backend requires CUDA")


def _token_ids(tokenizer: CausalTokenizer, text: str, *, specials: bool) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=specials, truncation=False)
    if not isinstance(encoded, dict) and not hasattr(encoded, "__getitem__"):
        raise TypeError("tokenizer output must expose input_ids")
    values = encoded["input_ids"]
    if isinstance(values, Tensor):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("single text produced multiple token sequences")
        values = values[0]
    if not isinstance(values, (tuple, list)):
        raise TypeError("tokenizer input_ids must be a sequence")
    result = tuple(int(x) for x in values)
    if not result or any(x < 0 for x in result):
        raise ValueError("empty or invalid tokenizer result")
    return result


def next_state_prompt(x_text: str) -> str:
    """Stable evaluation prompt. Not claimed to match BehR's TextWorld training."""
    if not isinstance(x_text, str) or not x_text:
        raise ValueError("nonempty pre-action X required")
    return (
        "Simulate one environment step from the agent-visible observation and "
        "the completed action. Output the next observation and action outcome "
        "in exactly the same canonical JSON schema.\n"
        "Before observation and performed action:\n"
        + x_text + "\n"
        "Next observation and action outcome:\n"
    )


class FrozenCausalLMTokenPort:
    """Expose full final-layer tokens from a genuinely frozen causal LM.

    All arrays returned are detached float32 copies; hidden-state comparisons
    are distinct from the model's trained next-token likelihood objective.
    Actual model loading is explicit. A mock model/tokenizer may be supplied
    for CPU-only contract tests and is never presented as a pretrained result.
    """

    def __init__(
        self, config: CausalLMConfig = CausalLMConfig(), *,
        model: nn.Module | None = None,
        tokenizer: CausalTokenizer | None = None,
    ) -> None:
        self.config = config
        self.spec = SPECS[config.candidate]
        if self.spec.kind != "causal_hidden_states":
            raise ValueError("AgentWorld is an external generative baseline, not a Qwen2 hidden-state port")
        if (model is None) != (tokenizer is None):
            raise ValueError("inject both model and tokenizer together")
        model_id = config.model_id or self.spec.model_id
        self.loaded_checkpoint_by_port = model is None
        if config.device == "cpu" and config.candidate in (
            Candidate.BEHR_TEXTWORLD, Candidate.BASE_QWEN25,
        ) and not config.allow_large_cpu and model is None:
            raise RuntimeError("7B CPU weight load blocked; use a GPU or explicit --allow-large-cpu")
        if config.device == "cuda" and not torch.cuda.is_available() and model is None:
            raise RuntimeError("CUDA unavailable; refusing to load large checkpoint")
        if model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
            except ImportError as exc:
                raise RuntimeError("Install: pip install -e '.[pretrained-encoder]'") from exc
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=config.revision, trust_remote_code=False,
            )
            kwargs: dict[str, object] = {
                "revision": config.revision,
                "trust_remote_code": False,
                "low_cpu_mem_usage": True,
                "torch_dtype": "auto" if config.device == "cuda" else torch.float32,
                "device_map": "auto" if config.device == "cuda" else "cpu",
            }
            if config.quantized_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:
                    raise RuntimeError("4-bit mode requires bitsandbytes support") from exc
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        assert model is not None and tokenizer is not None
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = config.revision or getattr(model.config, "_commit_hash", None)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self._dimension = int(getattr(self.model.config, "hidden_size"))
        if self._dimension < 16:
            raise ValueError("hidden size must be at least 16")
        self._cache: dict[str, np.ndarray] = {}
        self.max_observed_length = 0
        self.verified_length_count = 0
        self.backbone_forward_batches = 0
        self._nll_cache: dict[tuple[str, str], float] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _validated_ids(self, text: str, *, specials: bool) -> tuple[int, ...]:
        ids = _token_ids(self.tokenizer, text, specials=specials)
        self.max_observed_length = max(self.max_observed_length, len(ids))
        self.verified_length_count += 1
        if len(ids) > self.config.max_length:
            raise ValueError(
                f"input has {len(ids)} tokens; max_length={self.config.max_length}; "
                "silent truncation forbidden"
            )
        return ids

    def encode_many(self, texts: tuple[str, ...]) -> TokenStates:
        if not isinstance(texts, tuple) or not texts or any(type(t) is not str for t in texts):
            raise TypeError("nonempty tuple of immutable text inputs required")
        # Deterministic, unpadded forward passes avoid tokenizer padding effects
        # and make repeated state encodings cache-safe.
        fresh = tuple(dict.fromkeys(t for t in texts if t not in self._cache))
        for text in fresh:
            ids = self._validated_ids(text, specials=True)
            tensors = torch.tensor([ids], dtype=torch.long, device=self._model_device())
            with torch.inference_mode():
                output = self.model(
                    input_ids=tensors,
                    attention_mask=torch.ones_like(tensors),
                    output_hidden_states=True,
                    use_cache=False,
                )
            self.backbone_forward_batches += 1
            if not output.hidden_states:
                raise RuntimeError("model does not expose contextual hidden states")
            states = output.hidden_states[-1].detach().float().cpu().numpy()[0]
            if states.shape != (len(ids), self.dimension) or not np.isfinite(states).all():
                raise ValueError("invalid frozen model token output")
            self._cache[text] = states.copy()
        longest = max(len(self._cache[t]) for t in texts)
        tokens = np.zeros((len(texts),longest,self.dimension), dtype=np.float32)
        mask = np.zeros((len(texts),longest),dtype=bool)
        sentence = np.zeros((len(texts),self.dimension),dtype=np.float32)
        for i,text in enumerate(texts):
            row=self._cache[text]
            n=len(row)
            tokens[i,:n]=row
            mask[i,:n]=True
            last=row[-1].copy()
            sentence[i]=last / max(float(np.linalg.norm(last)), 1e-12)
        return TokenStates(tokens,mask,sentence)

    @torch.inference_mode()
    def conditional_nll(self, x_text: str, y_text: str) -> float:
        """Mean teacher-forced NLL of Y tokens conditioned on X + action.

        The target Y tokens may be supplied for *scoring*, never appended to
        the prefix. No target token is predicted from itself; next-token
        positions are shifted by one. Not a calibrated physical p(Y | do(X)).
        """
        if type(x_text) is not str or type(y_text) is not str or not y_text:
            raise TypeError("nonempty X/Y strings required")
        key=(x_text,y_text)
        if key in self._nll_cache:
            return self._nll_cache[key]
        prefix = self._validated_ids(next_state_prompt(x_text), specials=True)
        target = self._validated_ids(y_text, specials=False)
        total = prefix + target
        if len(total) > self.config.max_length:
            raise ValueError(
                f"teacher-forced transition has {len(total)} tokens; "
                f"max_length={self.config.max_length}; refusing truncation"
            )
        device=self._model_device()
        input_ids=torch.tensor([total],dtype=torch.long,device=device)
        out=self.model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),
                       use_cache=False)
        logits=out.logits[0,len(prefix)-1:len(total)-1].float()
        expected=torch.tensor(target,dtype=torch.long,device=logits.device)
        if logits.shape[0]!=len(expected) or logits.shape[1]<=int(expected.max()):
            raise ValueError("causal model logits incompatible with target tokens")
        value=float(F.cross_entropy(logits,expected,reduction="mean").cpu())
        if not isfinite(value):
            raise ValueError("nonfinite likelihood")
        self._nll_cache[key]=value
        return value


@dataclass(frozen=True)
class LikelihoodProbe:
    law: str
    seed: int
    pairs_examined: int
    skipped_identical_outcomes: int
    pair_accuracy: float | None
    mean_true_nll: float | None
    mean_wrong_nll: float | None
    mean_nll_margin: float | None


def evaluate_next_state_likelihood(
    port: FrozenCausalLMTokenPort,
    observed: tuple[TextRelationSample, ...],
    wrong: tuple[TextRelationSample, ...],
    *, law: str, seed: int, maximum: int | None = None,
) -> LikelihoodProbe:
    """Evaluation-only counterfactual pair test, never KAN training."""
    if len(observed)!=len(wrong):
        raise ValueError("true/alternative transitions must correspond one-to-one")
    if maximum is not None and (type(maximum) is not int or maximum<1):
        raise ValueError("maximum must be a positive integer")
    rows=[]
    skipped=0
    for left,right in zip(observed,wrong):
        if left.x_text!=right.x_text:
            raise ValueError("counterfactual altered the pre-action X")
        if left.y_text==right.y_text:
            skipped+=1
            continue
        rows.append((left.x_text,left.y_text,right.y_text))
        if maximum is not None and len(rows)>=maximum:
            break
    true=[]
    false=[]
    for x,y,n in rows:
        true.append(port.conditional_nll(x,y))
        false.append(port.conditional_nll(x,n))
    if not true:
        return LikelihoodProbe(law,seed,0,skipped,None,None,None,None)
    positive=np.asarray(true,dtype=float)
    negative=np.asarray(false,dtype=float)
    return LikelihoodProbe(
        law,seed,len(positive),skipped,
        float((positive<negative).mean()),
        float(positive.mean()),float(negative.mean()),
        float((negative-positive).mean()),
    )
