"""Phase 3 - 实验 1：EDA 量化选择偏差 / 曝光偏差（log_standard vs log_random）。

运行方式：
    .venv/bin/python scripts/01_eda_bias_analysis.py

观察重点：
1. 常规推荐流量（is_rand=0）与随机插播流量（is_rand=1）在点击率、长观率、完播比、负反馈上的巨大差距；
2. 曝光流行度集中度（Gini 系数 & Top-10% 视频曝光占比），直观看到“马太效应”；
3. 按 standard 曝光热度分桶（Cold / Mid / Head），观察冷门视频与热门视频在两种流量下的点击率反转/落差；
4. 用户活跃度（混杂因子 Confounder）在两种流量下的分布偏移。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kuairand_loader import load_kuairand_pure


def gini_coefficient(counts: np.ndarray) -> float:
    """计算曝光分布基尼系数（0=绝对均匀，越接近 1 越严重马太效应）。"""
    arr = np.sort(np.asarray(counts, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1.0) / n)


def main() -> None:
    bundle = load_kuairand_pure()
    std = bundle.log_standard.copy()
    rnd = bundle.log_random.copy()
    users = bundle.user_features.copy()

    print("=" * 78)
    print(f"[数据源] {bundle.source_desc}")
    if bundle.is_synthetic:
        print(
            "  提示：检测到 data/ 下尚无真实 KuaiRand-Pure CSV，当前运行的是严格对齐 Schema 的仿真数据。\n"
            "  放入真实 KuaiRand-Pure（解压至 data/KuaiRand-Pure/）后重新运行本脚本即可零改动切换！"
        )
    print("=" * 78)

    # 1. 核心反馈信号对比
    for df in (std, rnd):
        dur = df["duration_ms"].replace(0, np.nan)
        df["play_ratio"] = (df["play_time_ms"] / dur).clip(0, 2.0).fillna(0.0)

    metrics = ["is_click", "long_view", "play_ratio", "is_like", "is_follow", "is_hate"]
    summary = pd.DataFrame(
        {
            "log_standard (有偏常规流)": [std[m].mean() for m in metrics],
            "log_random (无偏随机流)": [rnd[m].mean() for m in metrics],
        },
        index=metrics,
    )
    summary["比值 (standard / random)"] = summary["log_standard (有偏常规流)"] / summary[
        "log_random (无偏随机流)"
    ].replace(0, np.nan)

    print("\n【1. 核心反馈指标对比：常规推荐流 vs 随机插播流】")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # 2. 视频曝光流行度集中度（Exposure / Popularity Bias）
    all_videos = np.union1d(std["video_id"].unique(), rnd["video_id"].unique())
    std_v_counts = std["video_id"].value_counts().reindex(all_videos, fill_value=0)
    rnd_v_counts = rnd["video_id"].value_counts().reindex(all_videos, fill_value=0)

    top10_k = max(1, int(0.10 * len(all_videos)))
    std_top10_share = std_v_counts.nlargest(top10_k).sum() / std_v_counts.sum()
    rnd_top10_share = rnd_v_counts.nlargest(top10_k).sum() / rnd_v_counts.sum()

    print("\n【2. 视频曝光集中度（量化 Popularity / Exposure Bias）】")
    print(f"  候选视频总数              : {len(all_videos)}")
    print(
        f"  log_standard              : Gini = {gini_coefficient(std_v_counts.values):.4f} | "
        f"Top 10% 热门视频占总曝光 = {std_top10_share * 100:.2f}% | "
        f"零曝光视频数 = {(std_v_counts == 0).sum()}"
    )
    print(
        f"  log_random (均匀插播)     : Gini = {gini_coefficient(rnd_v_counts.values):.4f} | "
        f"Top 10% 视频占总曝光     = {rnd_top10_share * 100:.2f}% | "
        f"零曝光视频数 = {(rnd_v_counts == 0).sum()}"
    )

    # 3. 按 standard 曝光热度分桶（Cold / Mid / Head），观察选择偏差对点击率的扭曲
    # 用 standard 中的曝光次数划分冷/中/热视频
    q1, q2 = np.quantile(std_v_counts.values, [0.40, 0.85])
    pop_bucket = pd.Series(index=all_videos, dtype="object")
    pop_bucket[std_v_counts <= q1] = "1_Cold (低曝光冷门)"
    pop_bucket[(std_v_counts > q1) & (std_v_counts <= q2)] = "2_Mid  (中曝光腰部)"
    pop_bucket[std_v_counts > q2] = "3_Head (高曝光热门)"

    std["pop_bucket"] = std["video_id"].map(pop_bucket)
    rnd["pop_bucket"] = rnd["video_id"].map(pop_bucket)

    bucket_tbl = pd.DataFrame(
        {
            "std_曝光占比": std["pop_bucket"].value_counts(normalize=True),
            "std_点击率": std.groupby("pop_bucket")["is_click"].mean(),
            "rand_曝光占比": rnd["pop_bucket"].value_counts(normalize=True),
            "rand_真实无偏点击率": rnd.groupby("pop_bucket")["is_click"].mean(),
        }
    ).sort_index()
    bucket_tbl["点击率虚高倍率(std/rand)"] = bucket_tbl["std_点击率"] / bucket_tbl["rand_真实无偏点击率"]

    print("\n【3. 冷/腰/热视频分层对比：有偏点击率 vs 随机流真实无偏点击率】")
    print(bucket_tbl.to_string(float_format=lambda x: f"{x:.4f}"))

    # 4. 用户活跃度混杂检查（Confounder Check）
    if "user_active_degree" in users.columns:
        u_deg = users.set_index("user_id")["user_active_degree"]
        std["active_degree"] = std["user_id"].map(u_deg).fillna("unknown")
        rnd["active_degree"] = rnd["user_id"].map(u_deg).fillna("unknown")

        u_tbl = pd.DataFrame(
            {
                "std_曝光占比": std["active_degree"].value_counts(normalize=True),
                "std_点击率": std.groupby("active_degree")["is_click"].mean(),
                "rand_曝光占比": rnd["active_degree"].value_counts(normalize=True),
                "rand_点击率": rnd.groupby("active_degree")["is_click"].mean(),
            }
        ).fillna(0.0)
        print("\n【4. 用户活跃度分层（典型 Confounder）：既影响被曝光机会，也影响点击率】")
        print(u_tbl.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n[核心结论]")
    print("  1) log_standard 的整体点击率/长观率远高于 log_random，因为线上策略只曝光了高分匹配+热门视频；")
    print("  2) log_standard 的曝光 Gini 系数极高（严重马太效应），而 log_random 几乎完全均匀（接近 RCT）；")
    print("  3) 活跃用户既在 log_standard 拿走更多曝光，自身点击率也更高——这就是教科书级的 Confounder！")


if __name__ == "__main__":
    main()
