import os
import sys
from .query_templates import *


def format_multichoice_question(row, conf_mode="verbal_numerical", choices = 4):
    if choices == 4: #  MMLU and GPQA
        if conf_mode == "sampling":
            return MULTI_CHOICE_4_SHARED_SAMPLING_TEMPLATE.format(**row)
        elif conf_mode == "verbal_numerical":
            return MULTI_CHOICE_4_COT_VERBAL_CONFIDENCE_TEMPLATE.format(**row)
        elif conf_mode == "verbal_linguistic":
            return MULTI_CHOICE_4_COT_HEDGING_TEMPLATE.format(**row)
        elif conf_mode == "logit_perplexity":
            return MULTI_CHOICE_4_COT_TEMPLATE.format(**row)
        elif conf_mode == "token_sar":
            return MULTI_CHOICE_4_COT_TEMPLATE.format(**row)
        elif conf_mode == "decisiveness_grading":
            return MULTI_CHOICE_TEMPLATE.format(**row)
        else:
            raise ValueError(f"Unknown conf_mode: {conf_mode}")
    elif choices == 0: # MMLU PRO without set placeholders
        options = dict(zip([chr(ord('A') + i) for i in range(len(row["options"]))], row["options"]))
        letters = ",".join(options.keys())
        options_str = "\n"
        for k, v in options.items():
            options_str += "{}) {}\n".format(k, v)
        if conf_mode == "sampling":
            return MULTI_CHOICE_NO_OPTIONS_SHARED_SAMPLING_TEMPLATE.format(Question = row["question"], Letters = letters) + options_str
        elif conf_mode == "verbal_numerical":
            return MULTI_CHOICE_NO_OPTIONS_COT_VERBAL_CONFIDENCE_TEMPLATE.format(Question = row["question"], Letters = letters) + options_str
        elif conf_mode == "verbal_linguistic":
            return MULTI_CHOICE_NO_OPTIONS_COT_HEDGING_TEMPLATE.format(Question = row["question"], Letters = letters) + options_str
        elif conf_mode == "logit_perplexity":
            return MULTI_CHOICE_NO_OPTIONS_COT_TEMPLATE.format(Question = row["question"], Letters = letters) + options_str
        elif conf_mode == "token_sar":
            return MULTI_CHOICE_NO_OPTIONS_COT_TEMPLATE.format(Question = row["question"], Letters = letters) + options_str
        elif conf_mode == "decisiveness_grading":
            return row["question"] + "\n" + options_str
        else:
            raise ValueError(f"Unknown conf_mode: {conf_mode}")
    else:
        raise ValueError(f"No query template for {choices}-choice multiple choice questions")
    

def format_open_ended_question(row, conf_mode="verbal_numerical"):
    if conf_mode == "sampling":
        return OPEN_ENDED_COT_SHARED_SAMPLING_TEMPLATE.format(Question=row.get("problem", ""))
    elif conf_mode == "verbal_numerical":
        return OPEN_ENDED_COT_VERBAL_CONFIDENCE_TEMPLATE.format(Question=row.get("problem", ""))
    elif conf_mode == "verbal_linguistic":
        return OPEN_ENDED_COT_HEDGING_TEMPLATE.format(Question=row.get("problem", ""))
    elif conf_mode == "logit_perplexity":
        return OPEN_ENDED_COT_TEMPLATE.format(Question=row.get("problem", ""))
    elif conf_mode == "token_sar":
        return OPEN_ENDED_COT_TEMPLATE.format(Question=row.get("problem", ""))
    else:
        raise ValueError(f"Unknown conf_mode: {conf_mode}")


# shared sampling
def shared_sampling_path(eval_type, model, conf_mode, num_examples, n_repeats):
    if not n_repeats:
        n_repeats = 0
    if not num_examples:
        regen_stored_path = f"LLM-Calibration-Study/cache/{eval_type}_shared_sampling_{model.split("/")[-1]}_full_{n_repeats}"
    else:
        regen_stored_path = f"LLM-Calibration-Study/cache/{eval_type}_shared_sampling_{model.split("/")[-1]}_{num_examples}_{n_repeats}"
    if conf_mode == "eval_all" and not os.path.exists(regen_stored_path):
        print("Run tests with '--conf_mode sampling' before running '--conf_mode eval_all'")
        sys.exit()
    return regen_stored_path

# individual sampling
def ind_sampling_path(eval_type, model, conf_mode, num_examples, n_repeats):
    if not n_repeats:
        n_repeats = 0
    if num_examples:
        regen_stored_path = f"LLM-Calibration-Study/cache/{eval_type}_{model.split("/")[-1]}_{conf_mode}_full_{n_repeats}"
    else:
        regen_stored_path = f"LLM-Calibration-Study/cache/{eval_type}_{model.split("/")[-1]}_{conf_mode}_{num_examples}_{n_repeats}"
    return regen_stored_path