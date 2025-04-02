import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

results = pd.read_csv("results/results.csv")

def plot_2d_results(results):
    df = results.rename(columns={"index": "model"}).copy()

    # 填充缺失 model_size
    df["model_size"] = df["model_size"].fillna(0)

    # 归一化 model_size 用于控制点大小
    min_size, max_size = 50, 800
    if df["model_size"].max() > 0:
        norm_size = min_size + (df["model_size"] / df["model_size"].max()) * (max_size - min_size)
        df["point_size"] = np.where(df["model_size"] == 0, min_size, norm_size)
    else:
        df["point_size"] = 200

    # 设置调色板
    families = df["model_family"].unique()
    palette = dict(zip(families, sns.color_palette("Set2", n_colors=len(families))))
    df["color"] = df["model_family"].map(palette)

    # 是否 CoT
    df["is_cot"] = df["model"].str.lower().str.contains("cot")
    df["marker"] = df["is_cot"].apply(lambda x: "^" if x else "o")

    plt.figure(figsize=(8, 5))

    # 绘制点
    for _, row in df.iterrows():
        plt.scatter(
            row["accuracy"],
            row["verbal_ece"],
            s=row["point_size"],
            color=row["color"],
            alpha=0.9,
            edgecolors="k",
            linewidth=0.5,
            marker=row["marker"],
            zorder=3
        )
        plt.text(
            row["accuracy"] + 0.002,
            row["verbal_ece"] + 0.002,
            row["model"],
            fontsize=8,
            zorder=4
        )

    # 画箭头：非cot -> cot，同 family 用同一色
    for _, row in df[~df["is_cot"]].iterrows():
        base_name = row["model"]
        cot_name = base_name + "-cot"
        if cot_name in df["model"].values:
            target = df[df["model"] == cot_name].iloc[0]
            plt.annotate(
                "",  # no text
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

    # 图例
    family_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=family,
                   markerfacecolor=color, markersize=9, markeredgecolor="k", linewidth=0)
        for family, color in palette.items()
    ]
    cot_handles = [
        plt.Line2D([0], [0], marker='^', color='k', label='CoT', linestyle='None', markersize=10),
        plt.Line2D([0], [0], marker='o', color='k', label='Non-CoT', linestyle='None', markersize=9),
    ]
    plt.legend(handles=family_handles + cot_handles, title="Legend", loc="upper right", fontsize=9, title_fontsize=10)

    plt.title("Model Performance on MMLU (Accuracy vs. Verbal ECE)", fontsize=14)
    plt.xlabel("Accuracy", fontsize=12)
    plt.ylabel("Verbal ECE", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("results/mmlu_results.pdf")
    plt.show()


if __name__ == "__main__":
    plot_2d_results(results)
