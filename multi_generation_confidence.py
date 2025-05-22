import pickle
import pandas as pd
import os
import re
from collections import Counter

import pandas as pd
import os
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

dir = "cache_multi_generation"
files = os.listdir(dir)

MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Answer\s*:​​​​​​",  # Korean invisible character
    r"উত্তর\s*:",
    r"उत्तर\s*:",
    r"উত্তরঃ",
    r"উত্তর\s*:",
    r"Antwort\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"답\s*:",
    r"答案\s*：",
    r"答案\s*:",
    r"答\s*：",
    r"答\s*:",
    r"答复\s*：",
    r"答曰\s*：",
    r"الإجابة:",
    r"الجواب:",
    r"إجابة:",
    r"الإجابة النهائية:",
    r"الإجابة الصحيحة:",
    r"الإجابة الصحيحة هي:",
    r"الإجابة هي:",
    r"الجواب النهائي:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"答え\s*:",
    r"答え\s*：",
    r"回答\s*:",
    r"回答\s*：",
    r"解答\s*:",
    r"Jawaban\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"Ìdáhùn\s*:",
    r"Idáhùn\s*:",
    r"Àmọ̀nà\s*:",
    r"Àdáhùn\s*:",
    r"Ànúgọ\s*:",
    r"Àṣàyàn\s*:",
]

MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)

def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """
    if response:
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
    return ""

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
def extract_answer(response_text: str):
    response_text = normalize_response(response_text)
    answer_patterns = [
        r"[Aa]nswer:?[\s]*[\n]*([A-J])",   
        r"[Aa]nswer:[\s]*[\n]*\(?([A-J])\)?", 
        r"[Aa]nswer:[\s]*[\n]*\[?([A-J])\]?",  
        r"[Aa]nswer:[\s]*[\n]*([A-J])[,)]",         
        r"[Aa]nswer:[\s]*[\n]*([A-J])\s*,?.*",
        r"Answer:\n([A-J])\nConfidence",         
        r"answer is\s*\[?\(?([A-J])\]?\)?",   
        r"answer should be\s*\[?\(?([A-J])\]?\)?",   
        r"best option is \(?([A-J])\)?",
        r"best match is option \(?([A-J])\)?",
        r"the closest is \(?([A-J])\)?",
        r"Answer:\n*^([A-J])$",
        r"^([A-J])$"
    ]
    # max_search_scope = len(response_text.splitlines())
    # for end in range(3, max_search_scope, 10):
    search_scope =  "\n".join(response_text.splitlines()[::-1])
    extracted_answer = None
    # Default answer extracrion from Simple Evals
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, search_scope)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1)).strip()
            if extracted_answer in "ABCDEFGHIJ":
                return extracted_answer
    # More complex extraction regex
    for pattern in answer_patterns:
        match = re.search(pattern, search_scope, re.IGNORECASE)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1)).strip()
            if extracted_answer in "ABCDEFGHIJ":
                return extracted_answer
        match = re.search(pattern, search_scope, re.MULTILINE)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1)).strip()
            if extracted_answer in "ABCDEFGHIJ":
                return extracted_answer
    return None
def process_single_response(lst):
    return [extract_answer(r) for r in lst]

def process_multi_responses(lst):
    return [[extract_answer(r) for r in r_ls] for r_ls in lst]
def calculate_ece(confidences, accuracies, n_bins=10) -> float:
    """
    Calculate the expected calibration error (ECE) given a list of confidence scores (0-1) and accuracy scores (0 or 1).
    """
    df = pd.DataFrame({"conf": confidences, "acc": accuracies}).dropna()

    if len(df) == 0:
        return None

    confidences = torch.tensor(df["conf"].tolist())
    accuracies = torch.tensor(df["acc"].tolist())
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
def calculate_adaptive_ece(confidences, accuracies, n_bins=10) -> float:
    """
    Calculate the Adaptive Expected Calibration Error (AECE) using equal-frequency binning.
    """
    # Ensure inputs are torch tensors

    df = pd.DataFrame({"conf": confidences, "acc": accuracies}).dropna()

    confidences = df["conf"].tolist()
    accuracies = df["acc"].tolist()

    if len(df) == 0:
        return None
    
    if len(set(confidences)) == 1:
        return abs(np.mean(confidences) - np.mean(accuracies))

    confidences = torch.tensor(confidences, dtype=torch.float)
    accuracies = torch.tensor(accuracies, dtype=torch.float)

    # Compute bin boundaries (equal number of samples per bin)
    sorted_conf = np.sort(confidences.detach().cpu().numpy())
    bin_boundaries = np.interp(
        np.linspace(0, len(sorted_conf), n_bins + 1),
        np.arange(len(sorted_conf)),
        sorted_conf
    )
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    adaptive_ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            # Ensure both are torch tensors
            diff = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            adaptive_ece += diff * prop_in_bin

    return adaptive_ece.item()
def safe_roc_auc_score(acc, conf):
    try:
        return roc_auc_score(acc, conf)
    except:
        return None
def mcq_majority_vote(answer_list: list[str]) -> tuple[str, float]:
    try:
        answer_list = [a for a in answer_list if a is not None]

        counter = Counter(answer_list)
        majority_answer, weight = counter.most_common(1)[0]

        weight /= len(answer_list)
        return majority_answer, weight
    except:
        return None, None


def mcq_consistency_confidence(first_answer: str, answer_list: list[str]) -> tuple[str, float]:
    answer_list = [a for a in answer_list if a != None]
    consistent_answers = [a for a in answer_list if a == first_answer]
    try:
        return len(consistent_answers) / len(answer_list)
    except:
        return None
    
for file in files:
    if file in os.listdir("cache"):
        print(dir + "/" + file)
        with open(dir + "/" + file, 'rb') as f:
            multi = pickle.load(f)
        with open("cache/" + file, "rb") as f:
            single = pickle.load(f)

        all_res = []
        for i in range(len(multi)):
            all_res += multi[i]["rep_responses"]

        ordered_groups = []
        for i in range(int(len(all_res) / 10)):
            response_group = []
            for j in range(i, len(all_res), int(len(all_res) / 10)):
                response_group.append(all_res[j])
            ordered_groups.append(response_group)

        prompt_messages = [eg["prompt_messages"] for eg in multi]
        correct_answer = [eg["answer"] for eg in multi]
        multi_responses = ordered_groups
        # multi_logprobs = [eg["rep_logprobs"] for eg in multi]
        single_response = [eg["response"] for eg in single]
        # single_logprobs = [eg["logprobs"] for eg in single]
        df  = pd.DataFrame({
            "prompt_messages": prompt_messages,
            "correct_answer": correct_answer,
            "single_answer": process_single_response(single_response),
            "multi_answers": process_multi_responses(multi_responses),
            "multi_responses": multi_responses,
            # "multi_logprobs": multi_logprobs,
            "single_response": single_response,
            # "single_logprobs": single_logprobs
        })
        df[["majority_answer", "majority_confidence"]] = df["multi_answers"].apply(
            lambda x: pd.Series(mcq_majority_vote(x))
        )
        df["majority_accuracy"] = df["majority_answer"] == df["correct_answer"]
        df["consistency_accuracy"] = df["single_answer"] == df["correct_answer"]
        df["consistency_confidence"] = df.apply(
            lambda row: mcq_consistency_confidence(row["single_answer"], row["multi_answers"]),
            axis=1
        )
        df.to_csv("semantic_results/" + file)


semantic_df = pd.DataFrame()
for file in os.listdir("semantic_results"):
    df = pd.read_csv("semantic_results/" + file)
    df = df.dropna()
    row = pd.DataFrame({
        "model": [file.replace("mmlu_pro_shared_sampling_", "").replace("_full_0", "").strip().capitalize()],
        
        "consistency_accuracy": df["consistency_accuracy"].mean(),
        "consistency_ece": calculate_ece(df["consistency_confidence"], df["consistency_accuracy"]),
        "consistency_adaptive_ece": calculate_adaptive_ece(df["consistency_confidence"], df["consistency_accuracy"]),
        "consistency_auroc": safe_roc_auc_score(df["consistency_accuracy"], df["consistency_confidence"]),

        "majority_accuracy": df["majority_accuracy"].mean(),
        "majority_ece": calculate_ece(df["majority_confidence"], df["majority_accuracy"]),
        "majority_adaptive_ece": calculate_adaptive_ece(df["majority_confidence"], df["majority_accuracy"]),
        "majority_auroc": safe_roc_auc_score(df["majority_accuracy"], df["majority_confidence"]),
    })
    semantic_df = pd.concat([semantic_df, row], ignore_index=True)
semantic_df.sort_values(by="model").to_csv("semantic_results/semantic_results.csv")
semantic_df.sort_values(by="model")
