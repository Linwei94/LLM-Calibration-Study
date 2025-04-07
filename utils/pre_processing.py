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
Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the option LETTER, where the LETTER is one of A,B,C,D; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanation.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

LLM_UNCERTAINTY_COT_TEMPLATE = """
Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Explanation: [insert step-by-step analysis here]\nAnswer and Confidence (0-100): [ONLY the option LETTER, where the LETTER is one of A,B,C,D], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly give me the reply according to this format, don't give me any other words.

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


def format_multichoice_question(row, conf_mode="verbal"):
    if conf_mode == "verbal":
        return LLM_UNCERTAINTY_TEMPLATE.format(**row)
    elif conf_mode == "verbal_cot":
        return LLM_UNCERTAINTY_COT_TEMPLATE.format(**row)
    else:
        raise ValueError(f"Unknown conf_mode: {conf_mode}")