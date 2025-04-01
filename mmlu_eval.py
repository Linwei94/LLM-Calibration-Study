"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re

import pandas

from . import common
from .common import (
    HTML_JINJA,
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    format_multichoice_question,
    normalize_extracted_answer,
    normalize_response,
)
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

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
    def __init__(self, num_examples: int | None = None, language: str = "EN-US"):
        if language != "EN-US":
            url = f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv"
        else:
            url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        df = pandas.read_csv(url)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

    def extract_answer(self, response_text: str) -> str | None:
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

    def extract_answer_and_confidence(self, response_text: str, options) -> tuple[str | None, float | None]:
        """
        Extracts the answer and confidence from the response text.
        The answer is extracted using a regex pattern, and the confidence is set to None.
        """
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
                if answer_option_value in option_value or \
                    option_value in answer_option_value:
                    answer_option_key = option_key
            
            # assert answer_option_key is not None, (match.group(0), answer_option_value, options)
            if answer_option_key is None:
                print(match.group(0), answer_option_value, options)
                # it returns an answer that does not belong to any of the option values
                return "Z", conf_scale
            
            return answer_option_key, conf_scale

        # Define five different regular expression patterns
        patterns_multi_choice = [
            r"^\s*([A-Z])\s*,\s*(\d+)%?", # A, 100%
            r"\s*([A-Z])\)\s*(\d+)", # C) 80
            r"\s*([A-Z])\)\s*.+?,\s*(\d+)%?", # B) currents and wind patterns, 100%
            r"\s*([A-Z])\)\s*.+?[\.，,]?\s*(\d+)%?", # B) currents and wind patterns. 100%
            r"(?:Answer and Confidence(?:\s*\([A-Z]\))?|Answer):\s*[\(\[]?([A-Z])[\)\]]?,?\s*(?:Confidence:\s*)?(\d+)",
            r"Answer and Confidence(?:\s*\([A-Z]\))?:\s*[\(\[]?([A-Z])[\)\]]?,?\s*(\d+)%?"
            r"Answer and Confidence\s*(?:\(0-100\))?:\s*[\(\[]?([A-Z])[\)\]]?[,]?\s*(\d+)%?",
            r"Answer and Confidence\s*(?:\(0-100\))?:\s*[\(\[]?([A-Z])[\)\]]?[,]?\s*[\(\[]?(\d+)%?[\(\[]?",
            r"Answer and Confidence\s*(?:\(0-100\))?:\s*[\(\[]?([A-Z])[\)\]]?[,.]?\s*.*[\(\[]?(\d+)%?[\(\[]?",
        ]
        # sometimes the LLM will directly output the answer rather than the associated option key
        patterns_multi_choice_without_option = [
            r"Answer and Confidence\s*(?:\(0-100\))?:\s*(.*?)\s*,\s*(\d+)%*"
        ]
        
        # [\(\[]?([A-Z])[\)\]]?  -> [\(\[]? matches optional ([
        # [\)\]]? matches optional )]
        # most appears in vicuna
        # Note: .* can match any character (except for a newline character) zero or more times
        patterns_multi_choice_werid = [
            r"Answer: [\(\[]?([A-Z])[\)\]]?[,.]?\s+Confidence level: (\d+%)",
            r"Answer: [\(\[]?([A-Z])[\)\]]?[,.]?.*\s+Confidence(?: level)?: (\d+%)",
            r"Answer:\s*[\(\[]?([A-Z])[\)\]]?[,.]?\s+Confidence level:\s*(\d+%)"
        ]

        patterns_and_postprocess_multi_choice = []
        patterns_and_postprocess_multi_choice.extend([(pat, default_postprocess_match) for pat in patterns_multi_choice])
        patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_without_option])
        patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_werid])

        # pre-process
        response_text = response_text.replace("(1-100)", "(0-100)")
        
        # Try each regular expression pattern in turn, until a match is found or all patterns have been tried
        is_match = False
        
        # task_type = "multi_choice"
        patterns_and_postprocess = patterns_and_postprocess_multi_choice
        
        answer, conf = None, None
        for pattern, match_processor in patterns_and_postprocess:
            match = re.search(pattern, response_text)
            if not match:
                continue
            answer, conf = match_processor(match)
            answer = None if (answer not in 'ABCD') and (answer != "place_holder") else answer # If a match is found, check whether the matched result meets the requirements
            if answer is not None and conf is not None:
                is_match = True
                break     
                    
        if not is_match:
            # If no match is found, print a message
            answer = None
            conf = None

        # extract answer using standard MMLU code
        extracted_answer_MMLU = None
        for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
            regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
            match = re.search(regex, response_text)
            if match:
                extracted_answer_MMLU = normalize_extracted_answer(match.group(1))
                break

        answer = answer if answer is not None else extracted_answer_MMLU

        return answer, conf

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(row), role="user"
                )
            ]
            response_text = normalize_response(sampler(prompt_messages))
            # Extract the answer from the response text
            extracted_answer, confidence = self.extract_answer_and_confidence(response_text, options={k: row[k] for k in ['A', 'B', 'C', 'D']})
            print(f"extracted_answer: {extracted_answer}, confidence: {confidence}")
            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                extracted_answer_confidence=confidence,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = subject2category.get(row["Subject"], "other")
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo, verbal_confidence=confidence
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
