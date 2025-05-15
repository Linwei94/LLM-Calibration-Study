from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """
    model: str
    get_logprobs: bool
    logprobs = None | list
    top_logprobs = None | list[dict]
    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None  # top-line metric
    metrics: dict[str, float] | None  # other metrics
    htmls: list[str]  # strings of valid HTML
    convos: list[MessageList]  # sampled conversations
    verbal_ece: float | None = None  # verbal confidence ECE


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None  # sampled conversation
    confidence: float | None = None
    category: str | None = None
    correct_answer: str | None = None
    extracted_answer: str | None = None
    semantic_entropy_answer: str | None = None
    response_token_length: int | None = 0
    verbal_numerical_confidence: float | None = 0
    logit_perplexity_confidence: float | None = 0
    verbal_linguistic_confidence: float | None = 0
    semantic_entropy: float | None = 0
    token_length: int | None = None
    


class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError
