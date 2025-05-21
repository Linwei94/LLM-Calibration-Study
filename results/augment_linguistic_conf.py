import os
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd

def get_confidence(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Define a regex pattern for "Extracted Answer Confidence"
    pattern = re.compile(r"Extracted Answer Confidence:\s([0-1\]*\.?[0-9]+)")

    # Search through the text in the HTML
    matches = pattern.findall(soup.get_text())

    # Print the extracted confidence values
    return np.array(matches, dtype=float)


# migrate legacy results to new result formats
all_csvs = ["/home/ivan/LLM-Calibration-Study/results/" + f for f in os.listdir("/home/ivan/LLM-Calibration-Study/results/") if f.endswith(".csv")]
results_dir = "/home/ivan/LLM-Calibration-Study/results/linguistic-judges/"
for html in os.listdir(results_dir):
    if "none_llama-4-maverick" in html.split("share")[-1].lower() and html.endswith("html"):
        model = html.split("_share")[0].replace("mmlu_pro_", "").replace("_verbal_linguistic", "")
        for csv in all_csvs:
            if model.lower() == csv.lower().split("mmlu_pro_")[-1].replace("_eval_all_none.csv", ""):
                print(f"Processing: {model}")
                df = pd.read_csv(csv)
                df["verbal_linguistic_confidence"] = get_confidence(results_dir + html)
                df.to_csv(csv)