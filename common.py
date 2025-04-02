import os
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Any

import io
import jinja2
import numpy as np
import requests
from tqdm import tqdm
import torch

import regex as re
from google.auth import default
from google.auth.transport.requests import Request

from .types import EvalResult, Message, SamplerBase, SingleEvalResult

from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from .sampler.hfmodel_sampler import HFChatCompletionSampler

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

QUERY_TEMPLATE_MULTICHOICE_WITH_CONFIDENCE = """
Read the following multiple choice question, analyze step by step, provide your answer and your confidence in this answer in the last line. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Explanation: [insert step-by-step analysis here]\nThe last line of your response should be of the following format: Answer: [ONLY the option LETTER where LETTER is one of ABCD; not a complete sentence], Confidence: [Your confidence level, please only include the numerical number in the range of 0-100]%\n```\nOnly give me the reply according to this format, don't give me any other words."

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

LLM_UNCERTAINTY_TEMPLATE = """
Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the option letter; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanation.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

LLM_UNCERTAINTY_COT_TEMPLATE = """
Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Explanation: [insert step-by-step analysis here]\nAnswer and Confidence (0-100): [ONLY the option letter; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly give me the reply according to this format, don't give me any other words.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()



LLM_UNCERTAINTY_TEMPLATE_WITHOUT_OPTIONS = """
Read the question, provide your answer and your confidence in this answer. 

Use the following format to answer:
```Answer: [Your answer], Confidence: [Your confidence level, please only include the numerical number in the range of 0-100]%```

Only the answer and confidence, don't give me the explanation.

{Question}
""".strip()


LLM_UNCERTAINTY_COT_TEMPLATE_WITHOUT_OPTIONS = """
Read the question, analyze step by step, provide your answer and your confidence in this answer. 

Use the following format to answer:
```Explanation: [insert step-by-step analysis here]\nAnswer: [Your answer], Confidence: [Your confidence level, please only include the numerical number in the range of 0-100]%```

Only give me the reply according to this format, don't give me any other words.

{Question}
""".strip()




ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    "Answer\s*:",
    "Answer\s*:​​​​​​",  # Korean invisible character
    "উত্তর\s*:",
    "उत्तर\s*:",
    "উত্তরঃ",
    "উত্তর\s*:",
    "Antwort\s*:",
    "답변\s*:",
    "정답\s*:",
    "답\s*:",
    "答案\s*：",
    "答案\s*:",
    "答\s*：",
    "答\s*:",
    "答复\s*：",
    "答曰\s*：",
    "الإجابة:",
    "الجواب:",
    "إجابة:",
    "الإجابة النهائية:",
    "الإجابة الصحيحة:",
    "الإجابة الصحيحة هي:",
    "الإجابة هي:",
    "الجواب النهائي:",
    "Respuesta\s*:",
    "Risposta\s*:",
    "答え\s*:",
    "答え\s*：",
    "回答\s*:",
    "回答\s*：",
    "解答\s*:",
    "Jawaban\s*:",
    "Réponse\s*:",
    "Resposta\s*:",
    "Jibu\s*:",
    "Idahun\s*:",
    "Ìdáhùn\s*:",
    "Idáhùn\s*:",
    "Àmọ̀nà\s*:",
    "Àdáhùn\s*:",
    "Ànúgọ\s*:",
    "Àṣàyàn\s*:",
]


EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Extracted Answer Confidence: {{ extracted_answer_confidence }}</p>
<p>Score: {{ score }}</p>
"""

def extract_confidence_from_response(response_text: str) -> str | None:
    """
    Extract the confidence score from the last line of the response string.
    Returns the confidence as a string if found, otherwise None.
    """
    last_line = response_text.strip().splitlines()[-1]

    # Try multiple patterns to handle various formats
    patterns = [
        r"Confidence:\s*(\d+(?:\.\d+)?)%?",                # e.g., Confidence: 80%
        r".*?,\s*Confidence:\s*(\d+(?:\.\d+)?)%?",          # e.g., Actor, Confidence: 80%
        r"Conf(?:idence)?\s*[:\-]?\s*(\d+(?:\.\d+)?)%?"     # e.g., Conf: 85 or Confidence-90%
    ]

    for pattern in patterns:
        match = re.search(pattern, last_line, re.IGNORECASE)
        if match:
            return match.group(1)

    return None



def format_multichoice_question(row, conf_mode="verbal"):
    if conf_mode == "verbal":
        return LLM_UNCERTAINTY_TEMPLATE.format(**row)
    elif conf_mode == "verbal_cot":
        return LLM_UNCERTAINTY_COT_TEMPLATE.format(**row)
    else:
        raise ValueError(f"Unknown conf_mode: {conf_mode}")


def check_equality(sampler: SamplerBase, expr1: str, expr2: str):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    response = sampler([dict(content=prompt, role="user")])
    return response.lower().strip() == "yes"


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    else:
        raise ValueError(f"Unknown {stat =}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """
    Aggregate results from multiple evaluations into a single EvalResult.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    htmls = []
    convos = []
    verbal_confidence_list = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        verbal_confidence_list.append(single_eval_result.verbal_confidence)
    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)

    # Calculate the verbalized ECE
    final_metrics['verbal_ece'] = calculate_ece(confidences=verbal_confidence_list, accuracies=name2values["score"])
    return EvalResult(
        score=final_metrics.pop("score", None), metrics=final_metrics, htmls=htmls, convos=convos
    )


def map_with_progress(f: callable, xs: list[Any], num_threads: int = 50):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    if os.getenv("debug"):
        return list(map(f, tqdm(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(tqdm(pool.imap(f, xs), total=len(xs)))


jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)
_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""


def message_to_html(message: Message) -> str:
    """
    Generate HTML snippet (inside a <div>) for a message.
    """
    return jinja_env.from_string(_message_template).render(
        role=message["role"], content=message["content"], variant=message.get("variant", None)
    )


jinja_env.globals["message_to_html"] = message_to_html


_report_template = """<!DOCTYPE html>
<html>
    <head>
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
</html>
"""


def make_report(eval_result: EvalResult) -> str:
    """
    Create a standalone HTML report from an EvalResult.
    """
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )


def make_report_from_example_htmls(htmls: list[str]):
    """
    Create a standalone HTML report from a list of example htmls
    """
    return jinja_env.from_string(_report_template).render(score=None, metrics={}, htmls=htmls)

def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )

def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )

def calculate_ece(confidences, accuracies, n_bins=15) -> float:
    """
    Calculate the expected calibration error (ECE) given a list of confidence scores and accuracy scores.
    """
    confidences = torch.tensor([i/100 for i in confidences]) # turn confidences to 0~1
    accuracies = torch.tensor(accuracies)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()



def url_to_fileobj(url: str, binary=False) -> Any:
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content) if binary else io.StringIO(response.text)


def get_model_dict(model_name: str):
    """
    Get the model dict from the model name.
    """
    models = {}
    if model_name == "gpt-4o-2024-11-20_assistant":
        models["gpt-4o-2024-11-20_assistant"] = ChatCompletionSampler(
            model="gpt-4o-2024-11-20",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )
    elif model_name == "gpt-4o-2024-11-20_chatgpt":
        models["gpt-4o-2024-11-20_chatgpt"] = ChatCompletionSampler(
            model="gpt-4o-2024-11-20",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        )
    elif model_name == "gpt-4o-mini":
        models["gpt-4o-mini"] = ChatCompletionSampler(
            model="gpt-4o-mini",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        )
    elif model_name == "o1":
        models["o1"] = OChatCompletionSampler(
            model="o1",
        )
    elif model_name == "o1-preview":
        models["o1-preview"] = OChatCompletionSampler(
            model="o1-preview",
        )
    elif model_name == "o1-mini":
        models["o1-mini"] = OChatCompletionSampler(
            model="o1-mini",
        )
    elif model_name == "o3-mini":
        models["o3-mini"] = OChatCompletionSampler(
            model="o3-mini",
        )
    elif model_name == "o3-mini_high":
        models["o3-mini_high"] = OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="high",
        )
    elif model_name == "o3-mini_low":
        models["o3-mini_low"] = OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="low",
        )
    elif model_name == "gpt-4-turbo-2024-04-09_assistant":
        models["gpt-4-turbo-2024-04-09_assistant"] = ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        )
    elif model_name == "gpt-4-turbo-2024-04-09_chatgpt":
        models["gpt-4-turbo-2024-04-09_chatgpt"] = ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        )
    elif model_name == "gpt-4o_assistant":
        models["gpt-4o_assistant"] = ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )
    elif model_name == "gpt-4o_chatgpt":
        models["gpt-4o_chatgpt"] = ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        )
    elif model_name == "gpt-4o-mini-2024-07-18":
        models["gpt-4o-mini-2024-07-18"] = ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )
    elif model_name == "gpt-4.5-preview-2025-02-27":
        models["gpt-4.5-preview-2025-02-27"] = ChatCompletionSampler(
            model="gpt-4.5-preview-2025-02-27",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )
    elif model_name == "claude-3-opus-20240229_empty":
        models["claude-3-opus-20240229_empty"] = ClaudeCompletionSampler(
            model="claude-3-opus-20240229",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        )
    elif model_name == "Llama-3.2-3B-Instruct":
        models["Llama-3.2-3B-Instruct"] = HFChatCompletionSampler(
                model="meta-llama/Llama-3.2-3B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Qwen2.5-3B-Instruct":
        models["Qwen2.5-3B-Instruct"] = HFChatCompletionSampler(
                model="Qwen/Qwen2.5-3B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Qwen2.5-7B-Instruct":
        models["Qwen2.5-7B-Instruct"] = HFChatCompletionSampler(
                model="Qwen/Qwen2.5-7B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Qwen2.5-13B-Instruct":
        models["Qwen2.5-13B-Instruct"] = HFChatCompletionSampler(
                model="Qwen/Qwen2.5-32B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Qwen2.5-32B-Instruct":
        models["Qwen2.5-32B-Instruct"] = HFChatCompletionSampler(
                model="Qwen/Qwen2.5-72B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Llama-3.1-405B-Instruct":
        models["Llama-3.1-405B-Instruct"] = HFChatCompletionSampler(
                model="meta-llama/Llama-3.1-405B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Llama-3.3-70B-Instruct": 
        models["Llama-3.3-70B-Instruct"] = HFChatCompletionSampler(
                model="meta-llama/Llama-3.3-70B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Llama-3.1-8B-Instruct": 
        models["Llama-3.1-8B-Instruct"] = HFChatCompletionSampler(
                model="meta-llama/Llama-3.1-8B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "DeepSeek-R1":
        models["DeepSeek-R1"] = HFChatCompletionSampler(
                model="deepseek-ai/DeepSeek-R1",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
        
    # add google cloud apis
    credentials, _ = default()
    auth_request = Request()
    credentials.refresh(auth_request)
    base_url = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi"
    
    models["google-llama-3.1-405b-instruct-maas"] = ChatCompletionSampler(
        model="meta/llama-3.1-405b-instruct-maas",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        base_url=base_url,
        api_key=credentials.token
    )
    models["google-llama-3.1-70b-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-3.1-70b-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=base_url,
            api_key=credentials.token
        )
    models["google-llama-3.1-8b-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-3.1-8b-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
            api_key=credentials.token
        )
    models["google-llama-3.3-70b-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-3.3-70b-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
            api_key=credentials.token
        )
    models["google-llama-3.1-8b-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-3.1-8b-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
            api_key=credentials.token
        )
        
        
    return models