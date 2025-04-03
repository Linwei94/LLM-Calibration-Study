import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import re
from bs4 import BeautifulSoup
from reliability_plots import reliability_plot, bin_strength_plot

def plot_benchmarks_vertical(results_path="plots/results.csv"):
    results = pd.read_csv(results_path)
    # 只保留需要n_samples=100的行
    results = results[results["n_samples"] == 100]
    benchmarks = results["benchmark"].unique()
    num_bench = len(benchmarks)

    fig, axes = plt.subplots(num_bench, 1, figsize=(8, 5 * num_bench), sharex=False, sharey=False)

    if num_bench == 1:
        axes = [axes]  # 保证 axes 可迭代

    all_families = results["model_family"].unique()
    palette = dict(zip(all_families, sns.color_palette("Set2", n_colors=len(all_families))))

    for ax, benchmark in zip(axes, benchmarks):
        df = results[results["benchmark"] == benchmark].copy()
        df["model_size"] = df["model_size"].fillna(0)

        # 归一化点大小
        min_size, max_size = 50, 800
        if df["model_size"].max() > 0:
            norm_size = min_size + (df["model_size"] / df["model_size"].max()) * (max_size - min_size)
            df["point_size"] = np.where(df["model_size"] == 0, min_size, norm_size)
        else:
            df["point_size"] = 200

        df["color"] = df["model_family"].map(palette)
        df["is_cot"] = df["model"].str.lower().str.contains("cot")
        df["marker"] = df["is_cot"].apply(lambda x: "^" if x else "o")

        # 绘制每个点
        for _, row in df.iterrows():
            ax.scatter(
                row["accuracy"],
                row["verbal_ece"],
                s=row["point_size"],
                color=row["color"],
                marker=row["marker"],
                edgecolors="k",
                linewidth=0.5,
                alpha=0.9,
                zorder=3
            )
            ax.text(
                row["accuracy"] + 0.002,
                row["verbal_ece"] + 0.002,
                row["model"],
                fontsize=8,
                zorder=4
            )

        # 绘制箭头：non-COT -> COT
        for _, row in df[~df["is_cot"]].iterrows():
            base_name = row["model"]
            cot_name = base_name + "-cot"
            if cot_name in df["model"].values:
                target = df[df["model"] == cot_name].iloc[0]
                ax.annotate(
                    "",
                    xy=(target["accuracy"], target["verbal_ece"]),
                    xytext=(row["accuracy"], row["verbal_ece"]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=row["color"],
                        lw=1.5,
                        alpha=0.8,
                    ),
                    zorder=2
                )

        ax.set_title(f"{benchmark}", fontsize=14)
        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_ylabel("Verbal ECE", fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', labelsize=10)

    # 构造统一图例
    family_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=family,
                   markerfacecolor=color, markersize=9, markeredgecolor="k", linewidth=0)
        for family, color in palette.items()
    ]
    cot_handles = [
        plt.Line2D([0], [0], marker='^', color='k', label='CoT', linestyle='None', markersize=10),
        plt.Line2D([0], [0], marker='o', color='k', label='Non-CoT', linestyle='None', markersize=9),
    ]

    fig.legend(
        handles=family_handles + cot_handles,
        title="Legend",
        loc="lower center",
        ncol=len(family_handles) + 2,
        fontsize=9,
        title_fontsize=10,
        bbox_to_anchor=(0.5, -0.01)
    )

    plt.suptitle("Model Performance by Benchmark", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92, hspace=0.35)
    plt.savefig("plots/all_benchmark_results.pdf", bbox_inches="tight")
    plt.show()

def read_html(result_html_path):
    with open(result_html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    confs, preds, labels = [], [], []
    results_sections = soup.find_all("h3", string="Results")
    for result in results_sections:
        correct_tag = result.find_next_sibling("p", string=re.compile("Correct Answer:"))
        extracted_tag = result.find_next_sibling("p", string=re.compile("Extracted Answer:"))
        conf_tag = result.find_next_sibling("p", string=re.compile("Extracted Answer Confidence:"))

        if correct_tag and extracted_tag and conf_tag:
            correct = re.search(r"Correct Answer:\s+([A-Z])", correct_tag.text)
            extracted = re.search(r"Extracted Answer:\s+([A-Z])", extracted_tag.text)
            conf = re.search(r"(\d+)", conf_tag.text)
            if correct and extracted and conf:
                labels.append(correct.group(1))
                preds.append(extracted.group(1))
                confs.append(int(conf.group(1)) / 100.0)

    return confs, preds, labels





def plot_reliability_diagram_from_html(result_html_path):
    confs, preds, labels = read_html(result_html_path)
    num_bins = 15

    # Plotting the reliability plot
    reliability_plot(confs, preds, labels, num_bins=num_bins)
    bin_strength_plot(confs, preds, labels, num_bins=num_bins)



if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Plotting functions")
    argparse.add_argument(
        "--file_path",
        type=str,
    )
    args = argparse.parse_args()
    # plot_benchmarks_vertical()
    plot_reliability_diagram_from_html(args.file_path)
    # Plotting the reliability plot
