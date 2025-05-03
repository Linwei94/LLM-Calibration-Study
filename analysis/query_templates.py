# 4-choice multiple choice question query template
# ---------------------------------------------------------------------
MULTI_CHOICE_4_SHARED_SAMPLING_TEMPLATE = """
Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\n
If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.\n
Use the following format to answer:\n
```Explanation: [insert short step-by-step analysis here]\n
Answer and Confidence (0-100): [ONLY the option LETTER, where the LETTER is one of A,B,C,D], [Your confidence level, please only include the numerical number in the range of 0-100]%```\n
Only give me the reply according to this format, don't give me any other words.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MULTI_CHOICE_4_COT_VERBAL_CONFIDENCE_TEMPLATE = """
Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\n
Use the following format to answer:\n
```Explanation: [insert short step-by-step analysis here]\n
Answer and Confidence (0-100): [ONLY the option LETTER, where the LETTER is one of A,B,C,D], [Your confidence level, please only include the numerical number in the range of 0-100]%```\n
Only give me the reply according to this format, don't give me any other words.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MULTI_CHOICE_4_COT_HEDGING_TEMPLATE = """
Read the question, analyze step by step, provide your answer.\n
If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.\n
Use the following format to answer:\n
```Explanation: [insert short step-by-step analysis here]\n
Answer: [ONLY the option LETTER, where the LETTER is one of A,B,C,D]```\n
The answer should be in the last line alone. Only give me the reply according to this format, don't give me any other words.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MULTI_CHOICE_4_COT_TEMPLATE = """
Read the question, analyze step by step, provide your answer.\n
Use the following format to answer:\n
```Explanation: [insert short step-by-step analysis here]\n
Answer: [ONLY the option LETTER, where the LETTER is one of A,B,C,D]```\n
The answer should be in the last line alone. Only give me the reply according to this format, don't give me any other words.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
# ---------------------------------------------------------------------

# 10-choice multiple choice question query template
# ---------------------------------------------------------------------
MULTI_CHOICE_NO_OPTIONS_SHARED_SAMPLING_TEMPLATE = """
Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\n
If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.\n
Use the following format to answer:\n
```Explanation: [insert short step-by-step analysis here]\n
Answer and Confidence (0-100): [ONLY the option LETTER, where the LETTER is one of A,B,C,D], [Your confidence level, please only include the numerical number in the range of 0-100]%```\n
Only give me the reply according to this format, don't give me any other words.

{Question}

""".strip()


MULTI_CHOICE_NO_OPTIONS_COT_VERBAL_CONFIDENCE_TEMPLATE = """
Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\n
Use the following format to answer:\n
```Explanation: [insert short step-by-step analysis here]\n
Answer and Confidence (0-100): [ONLY the option LETTER, where the LETTER is one of A,B,C,D], [Your confidence level, please only include the numerical number in the range of 0-100]%```\n
Only give me the reply according to this format, don't give me any other words.

{Question}

""".strip()


MULTI_CHOICE_NO_OPTIONS_COT_HEDGING_TEMPLATE = """
Read the question, analyze step by step, provide your answer.\n
If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.\n
Use the following format to answer:\n
```Explanation: [insert short step-by-step analysis here]\n
Answer: [ONLY the option LETTER, where the LETTER is one of A,B,C,D]```\n
The answer should be in the last line alone. Only give me the reply according to this format, don't give me any other words.

{Question}

""".strip()


MULTI_CHOICE_NO_OPTIONS_COT_TEMPLATE = """
Read the question, analyze step by step, provide your answer.\n
Use the following format to answer:\n
```Explanation: [insert short step-by-step analysis here]\n
Answer: [ONLY the option LETTER, where the LETTER is one of A,B,C,D]```\n
The answer should be in the last line alone. Only give me the reply according to this format, don't give me any other words.

{Question}

""".strip()
# ---------------------------------------------------------------------


# Open-ended question query template
# ---------------------------------------------------------------------
OPEN_ENDED_COT_VERBAL_CONFIDENCE_TEMPLATE = """
Read the question, analyze step by step, provide your answer and your confidence in this answer.\n

Use the following format to answer:
```Explanation: [insert short step-by-step analysis here]\nAnswer: [Your answer], Confidence: [Your confidence level, please only include the numerical number in the range of 0-100]%```\n

Only give me the reply according to this format, don't give me any other words.

{Question}
""".strip()


OPEN_ENDED_COT_SHARED_SAMPLING_TEMPLATE = """
Read the question, analyze step by step, provide your answer and your confidence in this answer.\n
If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.

Use the following format to answer:
```Explanation: [insert short step-by-step analysis here]\nAnswer: [Your answer], Confidence: [Your confidence level, please only include the numerical number in the range of 0-100]%```\n

Only give me the reply according to this format, don't give me any other words.

{Question}
""".strip()


OPEN_ENDED_COT_TEMPLATE = """
Read the question, analyze step by step, provide your answer. 

Use the following format to answer:
```Explanation: [insert short step-by-step analysis here]\nAnswer: [Your answer]```\n

Only give me the reply according to this format, don't give me any other words.

{Question}
""".strip()


OPEN_ENDED_COT_HEDGING_TEMPLATE = """
Read the question, analyze step by step, provide your answer. \n
If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.

Use the following format to answer:
```Explanation: [insert short step-by-step analysis here]\nAnswer: [Your answer]```\n

Only give me the reply according to this format, don't give me any other words.

{Question}
""".strip()
# ---------------------------------------------------------------------