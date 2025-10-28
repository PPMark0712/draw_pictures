import os, json
import numpy as np
import matplotlib.pyplot as plt


def draw_dist(data: list[float], x_label: str, output_fn: str, bins: int = 200) -> None:
    """绘制数据分布的概率密度函数"""
    n, bins_edges = np.histogram(data, bins=bins, density=True)
    bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bins_centers, n, linestyle='-', color='skyblue', label='Distribution')
    ax.set_title(f'Distribution of {x_label}', fontsize=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("density", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"save dist fig in {os.path.abspath(output_fn)}")


def draw_cdf(data: list[float], x_label: str, output_fn: str) -> None:
    """绘制数据的累积分布函数"""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(sorted_data, cdf, where='post', color='salmon', label='CDF')
    ax.set_title(f'Cumulative Distribution of {x_label}', fontsize=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"save cdf fig at {os.path.abspath(output_fn)}")


if __name__ == "__main__":
    output_path = "/mnt/data/kw/yyz/projects/CurriculumLearning/check/output/dist"
    os.makedirs(output_path, exist_ok=True)
    fn = "1.json"
    with open(fn, "r") as f:
        data = json.load(f)
    x_label = "score"
    draw_dist(data, x_label, os.path.join(output_path, f"dist_of_{x_label}.svg"))
    draw_cdf(data, x_label, os.path.join(output_path, f"cdf_of_{x_label}.svg"))