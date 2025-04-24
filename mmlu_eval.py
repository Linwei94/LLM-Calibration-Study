"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import pickle
import os
import pandas
import sys

from . import common
from .common import (
    HTML_JINJA,
    format_multichoice_question,
    normalize_response,
    extract_answer_and_confidence,
)
from .custom_types import Eval, EvalResult, SamplerBase, SingleEvalResult

from .utils.report_post_processing import *
from .utils.confidence_post_processing import *
from .utils.pre_processing import *

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


class MMLUEval(Eval):
    def __init__(self, decisiveness_grader: SamplerBase, num_examples: int | None = None, language: str = "EN-US", conf_mode: str = "verbal_vanilla", regenerate=True):
        if language != "EN-US":
            url = f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv"
        else:
            url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        df = pandas.read_csv(url)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.conf_mode = conf_mode
        self.regenerate = regenerate
        self.num_examples = num_examples
        self.cache_found = False
        self.decisiveness_grader = decisiveness_grader

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):

            sampling = 5
            extracted_answer = ""
            confidence = 0

            match self.conf_mode:
                
                case "verbal_numerical" | "verbal_numerical_shared_sampling":
                    if self.cache_found:
                        response = row["response"] 
                        top_logprobs = row["top_logprobs"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(row, conf_mode="verbal_numerical"), role="user"
                            )
                        ]
                        response = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response"] = response
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                        row["logit_perplexity"] = sampler.logit_perplexity

                    response_text = normalize_response(response)
                    logprobs = row["logprobs"]
                    # Extract the answer from the response text
                    extracted_answer, confidence = extract_answer_and_confidence("\n".join(response_text.splitlines()[-2:]), options={k: row[k] for k in ['A', 'B', 'C', 'D']})
                    if extracted_answer is None or extracted_answer not in "ABCD" or float(confidence) < 10:
                        extracted_answer, confidence = extract_answer_and_confidence(response_text, options={k: row[k] for k in ['A', 'B', 'C', 'D']})
                    confidence /= 100


                case "logit_perplexity" | "logit_perplexity_shared_sampling":
                    sampler.logprobs = True
                    if self.cache_found:
                        response = row["response"] 
                        top_logprobs = row["top_logprobs"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(row, conf_mode="logit_perplexity"), role="user"
                            )
                        ]
                        response = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response"] = response
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                        row["logit_perplexity"] = sampler.logit_perplexity

                    response_text = normalize_response(response)
                    confidence = row["logit_perplexity"]
                    logprobs = row["logprobs"]

                    extracted_answer, _ = extract_answer_and_confidence("\n".join(response_text.splitlines()[-2:]), options={k: row[k] for k in ['A', 'B', 'C', 'D']})
                    if extracted_answer is None or extracted_answer not in "ABCD":
                        extracted_answer, _ = extract_answer_and_confidence(response_text, options={k: row[k] for k in ['A', 'B', 'C', 'D']})
                    if extracted_answer is None or extracted_answer not in "ABCD":
                        extracted_answer = extract_mcq_answer("\n".join(response_text.splitlines()[-3:]), "mmlu")


                case "verbal_linguistic" | "verbal_linguistic_shared_sampling":
                    if self.cache_found:
                        response = row["response"] 
                        top_logprobs = row["top_logprobs"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(row, conf_mode="verbal_linguistic"), role="user"
                            )
                        ]
                        response = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response"] = response
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                        row["logit_perplexity"] = sampler.logit_perplexity

                    logprobs = row["logprobs"]
                    response_text = normalize_response(response)

                    extracted_answer, _ = extract_answer_and_confidence("\n".join(response_text.splitlines()[-2:]), options={k: row[k] for k in ['A', 'B', 'C', 'D']})
                    if extracted_answer is None or extracted_answer not in "ABCD":
                        extracted_answer, _ = extract_answer_and_confidence(response_text, options={k: row[k] for k in ['A', 'B', 'C', 'D']})
                    if extracted_answer is None or extracted_answer not in "ABCD":
                        extracted_answer = extract_mcq_answer("\n".join(response_text.splitlines()[-3:]), "mmlu")
                    
                    confidence = decisiveness_score(self.decisiveness_grader, format_multichoice_question(row, "verbal_linguistic"), response_text)

                case "sampling":
                    sampler.logprobs = True
                    prompt_messages = [
                        sampler._pack_message(
                            content=format_multichoice_question(row, conf_mode="sampling"), role="user"
                        )
                    ]
                    response = sampler(prompt_messages)
                    row["prompt_messages"] = prompt_messages
                    row["response"] = response
                    row["logprobs"] = sampler.logprobs
                    row["top_logprobs"] = sampler.top_logprobs
                    row["logit_perplexity"] = sampler.logit_perplexity
                    return

                case _:
                    raise Exception(f"Unrecognized confidence type: {self.conf_mode}")

            print(f"extracted_answer: {extracted_answer}, confidence: {confidence}")
            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            category = subject2category.get(row["Subject"], "other")
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                extracted_answer_confidence=confidence,
                subject=category,
                logprobs = logprobs,
                conf_mode = self.conf_mode
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo, confidence=float(confidence)
            )


        # Run evaluation and collect results
        if self.conf_mode in ["sampling", "eval_all"] or "_shared_sampling" in self.conf_mode:
            self.regenerate = True
            regen_stored_path = shared_sampling_path("mmlu", sampler.model, self.conf_mode, self.num_examples, None)
        else:
            regen_stored_path = ind_sampling_path("mmlu", sampler.model, self.conf_mode, self.num_examples, None)
            

        if self.regenerate:
            if os.path.exists(regen_stored_path):
                print("Fetching from cache")
                with open(regen_stored_path, 'rb') as f:
                    self.examples = pickle.load(f)
                self.cache_found = True
                results = common.map_with_progress(fn, self.examples)
            else:
                print("No cache")
                results = common.map_with_progress(fn, self.examples)
                with open(regen_stored_path, 'wb') as f:
                    pickle.dump(self.examples, f)
        else:
            results = common.map_with_progress(fn, self.examples)

        if self.conf_mode == "sampling":
            with open(regen_stored_path, 'wb') as f:
                pickle.dump(self.examples, f)
            print(f"Shared sampling complete and saved to: {regen_stored_path}")
            sys.exit()

        return common.aggregate_results(results)
