import numpy as np

import regex as re
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

from ..custom_types import SamplerBase
from  .query_templates import *
import torch
from sentence_transformers import CrossEncoder
from huggingface_hub import login
import os


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


def check_equality(sampler: SamplerBase, expr1: str, expr2: str):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    response = sampler([dict(content=prompt, role="user")])
    return response.lower().strip() == "yes"


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

def mmlu_regex_extract_response(response_text):
    response_text = normalize_response(response_text)
    extracted_answer = None
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break
    return extracted_answer

def gpqa_regex_extract_response(response_text):
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    return extracted_answer

def mmlu_pro_regex_extract_response(response_text):
    def extract_final(text):
        pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return None
    def extract_again(text):
        match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
        if match:
            return match.group(1)
        else:
            return extract_final(text)
    def extract_answer(text):
        pattern = r"[answer is] \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return extract_again(text)
    return extract_answer(response_text)


mcq_regex_extractors = {"mmlu": mmlu_regex_extract_response, "gpqa": gpqa_regex_extract_response, "mmlu_pro": mmlu_pro_regex_extract_response}
def extract_mcq_answer(response_text: str, benchmark = "gpqa") -> str | None:
    """
    Extracts the answer from the response text.
    The answer is extracted using a regex pattern.
    """
    extracted_answer = None
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break
    if extracted_answer == None or extracted_answer.upper() not in "ABCDEFGHIJ":
        response_text = response_text.strip().splitlines()[-1]
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break
    if extracted_answer == None or extracted_answer.upper() not in "ABCDEFGHIJ":
        extracted_answer = mcq_regex_extractors[benchmark](response_text)
    return extracted_answer

def extract_answer_and_confidence(response_text: str, options) -> tuple[str | None, float | None]:
    # single-line format
    patterns_multi_choice = [
        r"[Aa]nswer:\s*([A-J])\s*\n*\s*[Cc]onfidence:\s*(\d{1,3})%?",
        r"[Aa]nswer:\s*([A-J])\s*[\n\r]*+[Cc]onfidence(?:\s*\(0-100\))?:\s*(\d{1,3})%?",
        r"[Aa]nswer:\s*([A-J])\s*[\n\r]*+[Cc]onfidence(?:\s*\(0-100\))?:\s*(\d+)%?",
        r"Answer:\s*([A-J])\s*[\n\r]+Confidence(?:\s*\(0-100\))?:\s*(\d{1,3})%",
        r"The best answer is (\w): [\d.]+, (\d+)%?\.",
        r"([A-J]):\s*[^,]+,\s*(\d+)%?",
        r"^\s*([A-J])\)\s*.*\n\s*(\d{1,3})\s*$"
        r"^\s*([A-J])\)\s*.*\n\s*(\d+)%?"
        r"^\s*([A-J])\)\s+.*?(\d{1,3})\s*$",
        r"^\s*([A-J])\s*,\s*(\d+)%?",
        r"\s*([A-J])\)\s*(\d+)",
        r"\s*([A-J])\)\s*.+?,\s*(\d+)%?",
        r"\s*([A-J])\)\s*.+?[\.\uFF0C,]?\s*(\d+)%?",
        r"Answer and Confidence\s*\(0-100\):\s*[\(\[]?([A-J]+)[\)\]]?,\s*(\d+)%?",
        r"Answer and Confidence \(0-100\): \s*[\(\[]?([A-J]+)[\)\]]?,\s*(\d+)%?",
        r"(?:Answer and Confidence(?:\s*\([A-J]\))?|Answer):\s*[\(\[]?([A-J])\)?\]?[,]?\s*(?:Confidence:\s*)?(\d+)",
        r"The best answer is ([A-J]):\s*[\d.]+,\s*(\d+)%?",
        r"Answer: [\(\[]?([A-J])\)?\]?[,.]?\s+Confidence(?: level)?:\s*(\d+)%?",
        r"The best answer is ([A-J]),\s*(\d+)%?",
        r"The best answer is ([A-J]):\s*(\d+)%?",
        r"The best answer is ([A-J]):\.?\s,*(\d+)%?",
        r"The best answer is ([A-J]):\s*(\d+)%?",
        r"The best answer is ([A-J]).\s*(\d+)%?",
    ]
    # single-line format
    patterns_multi_choice_without_option = [
        r"Answer and Confidence\s*(?:\(0-100\))?:\s*(.+?),\s*(\d+)%?"
        r"Answer and Confidence (100):\s*(.+?),\s*(\d+)%?"
    ]
    # single-line format
    patterns_multi_choice_weird = [
        r"Answer: [\(\[]?([A-J])\)?\]?[,.]?\s+Confidence level: (\d+)%",
        r"Answer: [\(\[]?([A-J])\)?\]?[,.]?.*\s+Confidence(?: level)?: (\d+)%",
        r"Answer:\s*[\(\[]?([A-J])\)?\]?[,.]?\s+Confidence level:\s*(\d+)%",
    ]
    multi_line_patters = [
        r"Answer:\s*([A-Ja-j])\s*\n*\s*Confidence(?:\s*\(0-100\))?\s*:\s*(\d{1,3})%?"
        r"Answer:\s*([A-Ja-j])\s*\n*\s*\(0-100\)(?:\s*\(0-100\))?\s*:\s*(\d{1,3})%?"
    ]
    
    patterns = multi_line_patters + patterns_multi_choice + patterns_multi_choice_without_option + patterns_multi_choice_weird
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            confidence = int(match.group(2))
            if 0 <= confidence <= 100 and answer in "ABCDEFGHIJ":
                return answer, confidence
    
    for pattern in patterns:
        match = re.search(pattern, "\n".join(response_text.splitlines()[-5:]), re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            confidence = int(match.group(2))
            if 0 <= confidence <= 100 and answer in "ABCDEFGHIJ":
                return answer, confidence
    return "Z", 0


def consolidated_answer_extraction(benchmark: str, response_text: str, row: dict | None = None, choices_dict: dict | None = None , with_verbal_confidence: bool = False) -> tuple[str, float] | str:
    """a single function that handles answer extraction for different benchmarks

    Args:
        benchmark (str): One of mmly_pro, mmlu, gpqa, simpleqa
        response_text (str): Processed response text from the sampler
        row (dict): A row from an example
        choices_dict (dict): Choice from an example specifically for GPQA
        with_verbal_confidence (bool, optional): Whether the response contains verbal confidence to extract. Defaults to False.

    Returns:
        tuple[str, float] | str: return answer and 
    """
    extracted_answer = None
    confidence = 0
    match benchmark:
        case "mmlu_pro":
            option_letters = "ABCDEFGHIJ"
            options = dict(zip([chr(ord('A') + i) for i in range(len(row["options"]))], row["options"]))
        case "mmlu":
            option_letters = "ABCD"
            options = {k: row[k] for k in ['A', 'B', 'C', 'D']}
        case "gpqa":
            if choices_dict is None:
                raise ValueError("choices_dict not specified for GPQA")
            option_letters = "ABCD"
            options = {k: choices_dict[k] for k in "ABCD"}

    if with_verbal_confidence:
        extracted_answer, confidence = extract_answer_and_confidence("\n".join(response_text.splitlines()[-3:]), options=options)
        if extracted_answer is None or extracted_answer not in option_letters or float(confidence) < 10:
            extracted_answer, confidence = extract_answer_and_confidence(response_text, options=options)
        return extracted_answer, confidence/100
    
    else:
        extracted_answer, _ = extract_answer_and_confidence("\n".join(response_text.splitlines()[-5:]), options=options)
        if extracted_answer is None or extracted_answer not in option_letters:
            extracted_answer, _ = extract_answer_and_confidence(response_text, options=options)
        if extracted_answer is None or extracted_answer not in option_letters:
            extracted_answer = extract_mcq_answer("\n".join(response_text.splitlines()[-10:]), benchmark)

        return extracted_answer