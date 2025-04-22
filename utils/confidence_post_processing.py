import numpy as np

import random
import regex as re
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from ..custom_types import SamplerBase


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

def extract_answer(response_text: str) -> str | None:
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
    return extracted_answer

def extract_answer_and_confidence(response_text: str, options) -> tuple[str | None, float | None]:
    def default_postprocess_match(match) -> tuple[str, str]:
        assert match is not None
        option_key, conf_scale = match.group(1), match.group(2)
        return option_key, conf_scale

    def postprocess_match_without_option(match) -> tuple[str, str]:
        assert match is not None
        answer_option_value = match.group(1).strip()
        conf_scale = match.group(2)
        answer_option_key = None
        for option_key, option_value in options.items():
            option_value = option_value.strip().strip(".").lower()
            answer_option_value = answer_option_value.strip().strip(".").lower()
            if answer_option_value in option_value or option_value in answer_option_value:
                answer_option_key = option_key

        if answer_option_key is None:
            print(match.group(0), answer_option_value, options)
            return "Z", conf_scale

        return answer_option_key, conf_scale

    response_text = response_text.replace("(1-100)", "(0-100)")
    lines = response_text.strip().splitlines()[::-1]  # reverse the lines

    # Regular expressions
    patterns_multi_choice = [
        r"^\s*([A-Z])\)\s*.*\n\s*(\d{1,3})\s*$"
        r"^\s*([A-Z])\)\s*.*\n\s*(\d+)%?"
        r"^\s*([A-Z])\)\s+.*?(\d{1,3})\s*$",
        r"^\s*([A-Z])\s*,\s*(\d+)%?",
        r"\s*([A-Z])\)\s*(\d+)",
        r"\s*([A-Z])\)\s*.+?,\s*(\d+)%?",
        r"\s*([A-Z])\)\s*.+?[\.\uFF0C,]?\s*(\d+)%?",
        r"Answer and Confidence\s*\(0-100\):\s*[\(\[]?([A-Z0-9]+)[\)\]]?,\s*(\d+)%?",
        r"(?:Answer and Confidence(?:\s*\([A-Z]\))?|Answer):\s*[\(\[]?([A-Z])\)?\]?[,]?\s*(?:Confidence:\s*)?(\d+)",
        r"Answer: [\(\[]?([A-Z])\)?\]?[,.]?\s+Confidence(?: level)?:\s*(\d+)%?",
    ]

    patterns_multi_choice_without_option = [
        r"Answer and Confidence\s*(?:\(0-100\))?:\s*(.+?),\s*(\d+)%?"
    ]

    patterns_multi_choice_weird = [
        r"Answer: [\(\[]?([A-Z])\)?\]?[,.]?\s+Confidence level: (\d+)%",
        r"Answer: [\(\[]?([A-Z])\)?\]?[,.]?.*\s+Confidence(?: level)?: (\d+)%",
        r"Answer:\s*[\(\[]?([A-Z])\)?\]?[,.]?\s+Confidence level:\s*(\d+)%"
    ]

    patterns_and_postprocess_multi_choice = []
    patterns_and_postprocess_multi_choice.extend([(pat, default_postprocess_match) for pat in patterns_multi_choice])
    patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_without_option])
    patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_weird])

    answer, conf = None, None
    for line in lines:
        for pattern, match_processor in patterns_and_postprocess_multi_choice:
            match = re.search(pattern, line)
            if match:
                answer, conf = match_processor(match)
                if answer in 'ABCD' or answer == 'place_holder':
                    if float(conf) > 100:
                        conf = 100
                    if float(conf) < 0 or conf==None:
                        conf = 0
                    return answer, float(conf)

    
    # if the extraction answer is None or confidence is None, we try to extract the answer using the default method   
    if answer == None or conf == None:
        # answer is one of A, B, C, D ramdomly
        answer = random.choice(['A', 'B', 'C', 'D'])
        conf = 0
    if float(conf) > 100:
        conf = 100
    if float(conf) < 0:
        conf = 0
    return answer, float(conf)


# ------------------------------------------------------------------------------------------------------
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

mcq_regex_extractors = {"mmlu": mmlu_regex_extract_response, "gpqa": gpqa_regex_extract_response}

# Semantic-based Confidence
# ------------------------------------------------------------------------------------------------------
def get_semantic_clusters(multi_response):
    lnll_lst = [(x)[1] for x in multi_response]
    response_list = [x[0] for x in multi_response]
    distance_threshold = 0.3
    model_name="all-MiniLM-L6-v2"
    embeddings = SentenceTransformer(model_name).encode(response_list)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
    labels = clustering.fit_predict(embeddings)
    return response_list, lnll_lst, labels


def get_mcq_clusters(multi_response, test = "mmlu"):
    lnll_lst = [(x)[1] for x in multi_response]
    response_list = [x[0] for x in multi_response]
    choice_map = dict()
    choice_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    labels = [choice_map.get(mcq_regex_extractors[test](c), 0) for c in response_list]
    return response_list, lnll_lst, labels


def empirical_semantic_confidence(lnll_lst, response_list, labels):
    counts = Counter(labels)
    opt_cluster, opt_conf = max([(int(cluster_id), count/sum(counts.values())) for cluster_id, count in counts.items()], key=lambda x: x[1])
    optimal_response, index = max([(response_list[i], i) for i, label in enumerate(labels) if label == opt_cluster], key=lambda x: x[1])
    return optimal_response, opt_conf, index
# ------------------------------------------------------------------------------------------------------


# faithful response uncertainty
# ------------------------------------------------------------------------------------------------------
def confidence_by_contradiction(sampler, original_response, candidate_sample):
    contradiction_prompt = """I will give you a question and two candidate answers. Extract their assertions and determine whether the two answers are contradicting. If one of the Candidate Answers is a canned response (as in, avoids answering the question), then the verdict should be "no contradiction".\n"""
    vanilla_prompt = """Answer the following question using a succinct (at most one sentence) and full answer\n"""
    def is_contradicting(sampler, original_response: str, candidate_response: str) -> bool:
        msg = [{"role":"user","content":vanilla_prompt + " " + contradiction_prompt + f" Candidate Answer 1: {original_response} \nCandidate Answer 2: {candidate_response}" + """ Return 'true' for contradiction or 'false' for No Contradiction'"""}]
        verdict = sampler(msg)[0].lower()
        if 'true' in verdict:
            return True
        else:
            return False
    contradictions = [is_contradicting(sampler, original_response, c) for c in candidate_sample]
    return float(1 - np.mean(contradictions))

def decisiveness_score(sampler, question, response) -> tuple[str, float]:
    dec_prompt = """You will be given a question and a proposed answer. Your task is to extract any assertions made by the proposed answer; for each assertion, you should output a decisiveness score: a number in [0,1] representing how likely you think this assertion is to be true, judged solely based on the provided answer. You should pay special attention to the usage of any hedging modifiers in the original answer, used to convey uncertainty in the truthfulness of the assertion. If the proposed answer punts the question, the extracted assertion should be the empty string and the decisiveness score should be 1.0.\n"""
    vanilla_prompt = """Answer the following question using a succinct (at most one sentence) and full answer\n"""
    msg = [{"role":"user","content":vanilla_prompt + " " + dec_prompt  + f" Question: {question} \nProposed answer: {response}.\n Only outputs each assertion after 'Assertion:',  and decisiveness score, after 'Decisiveness score: ' in two lines."}]
    score_pattern = r"[Dd]ecisiveness [Ss]core:\s*([0-9]*\.?[0-9]+)"
    assertion_pattern = r"(?i)assertion:\s*(.*)"
    verdict = sampler(msg)[0]
    print(verdict)
    scores = re.findall(score_pattern, verdict)
    assertions = re.findall(assertion_pattern, verdict)
    verdict = verdict.split("\n")
    decisiveness_scores = [float(score) for score in scores]
    if len(decisiveness_scores) > 0:
        return decisiveness_scores[0]
    return 1
# ------------------------------------------------------------------------------------------------------
