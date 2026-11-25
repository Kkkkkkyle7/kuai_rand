"""KuaiRand-Pure 数据加载器（支持真实 CSV 与严格对齐 Schema 的合成回退）。

优先读取 `data/KuaiRand-Pure/data/` 或 `data/` 目录下的真实 KuaiRand-Pure CSV 文件；
如果本机尚未下载真实数据，会自动生成一份严格对齐 KuaiRand-Pure 字段与因果生成机制
（暴露偏差、流行度偏斜、用户活跃度混杂、12 维反馈信号、is_rand=0/1）的合成数据集，
确保所有 Phase 3 脚本开箱即跑，且一旦放入真实数据即可无缝切换。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


@dataclass
class KuaiRandBundle:
    log_standard: pd.DataFrame
    log_random: pd.DataFrame
    user_features: pd.DataFrame
    video_features: pd.DataFrame
    is_synthetic: bool
    source_desc: str


def _find_real_data_dir() -> Optional[Path]:
    """在 data/ 目录下探测真实 KuaiRand-Pure CSV 所在路径。"""
    candidates = [
        DATA_DIR / "KuaiRand-Pure" / "data",
        DATA_DIR / "KuaiRand-Pure",
        DATA_DIR,
    ]
    for base in candidates:
        if not base.exists():
            continue
        rand_files = list(base.glob("log_random_4_22_to_5_08*.csv"))
        std_files = list(base.glob("log_standard_4_22_to_5_08*.csv"))
        if rand_files and std_files:
            return base
    return None


def _load_real_kuairand_pure(
    base_dir: Path, max_std_rows: Optional[int] = 300_000, max_rand_rows: Optional[int] = 300_000
) -> KuaiRandBundle:
    """加载真实 KuaiRand-Pure 数据文件。"""
    std_file = sorted(base_dir.glob("log_standard_4_22_to_5_08*.csv"))[0]
    rand_file = sorted(base_dir.glob("log_random_4_22_to_5_08*.csv"))[0]
    user_file = sorted(base_dir.glob("user_features*.csv"))[0]
    video_file = sorted(base_dir.glob("video_features_basic*.csv"))[0]

    log_std = pd.read_csv(std_file, nrows=max_std_rows)
    log_rand = pd.read_csv(rand_file, nrows=max_rand_rows)
    user_feat = pd.read_csv(user_file)
    video_feat = pd.read_csv(video_file)

    return KuaiRandBundle(
        log_standard=log_std,
        log_random=log_rand,
        user_features=user_feat,
        video_features=video_feat,
        is_synthetic=False,
        source_desc=f"REAL KuaiRand-Pure ({base_dir})",
    )


def _generate_synthetic_kuairand_pure(
    n_users: int = 3_000,
    n_videos: int = 1_500,
    n_std_logs: int = 80_000,
    n_rand_logs: int = 40_000,
    seed: int = 42,
) -> KuaiRandBundle:
    """生成严格对齐 KuaiRand-Pure Schema 与因果结构（曝光偏差 + 混杂）的仿真数据。

    核心因果生成机制（DGP）：
    1. 用户有隐藏兴趣向量 + 活跃度分层（user_active_degree）；
    2. 视频有隐藏题材向量 + 真实质量（true_quality） + 外部流行度（popularity，
       故意设计一批“冷门但优质”的视频，用于验证 Naive 模型如何系统性低估冷门好视频）；
    3. 真实无偏偏好 P(Y=1 | u, i) 由“兴趣匹配 + 视频真实质量 + 用户活跃度”决定；
    4. log_standard（is_rand=0）由有偏线上策略曝光：严重偏向高流行度视频 + 高匹配/高活用户；
    5. log_random（is_rand=1）从全部候选视频中均匀随机抽样曝光（物理切断 X -> O 混杂路径）。
    """
    rng = np.random.default_rng(seed)

    # 1. 用户侧特征（对齐 user_features_pure.csv 核心字段）
    latent_dim = 4
    user_emb = rng.normal(0, 1, size=(n_users, latent_dim))
    # 活跃度：0=low, 1=middle, 2=high, 3=full
    active_probs = [0.20, 0.30, 0.30, 0.20]
    active_code = rng.choice([0, 1, 2, 3], size=n_users, p=active_probs)
    degree_map = {0: "low_active", 1: "middle_active", 2: "high_active", 3: "full_active"}
    active_effect = np.array([-0.55, -0.10, 0.30, 0.65])[active_code]
    register_days = np.clip(rng.exponential(scale=400, size=n_users).astype(int) + 10, 10, 2500)
    follow_user_num = np.clip((rng.lognormal(mean=2.5 + 0.5 * active_code, sigma=0.8)).astype(int), 0, 5000)
    fans_user_num = np.clip((rng.lognormal(mean=1.5 + 0.4 * active_code, sigma=1.1)).astype(int), 0, 50000)

    user_features = pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "user_active_degree": [degree_map[c] for c in active_code],
            "active_code": active_code,
            "register_days": register_days,
            "follow_user_num": follow_user_num,
            "fans_user_num": fans_user_num,
            "u_emb_0": user_emb[:, 0],
            "u_emb_1": user_emb[:, 1],
            "u_emb_2": user_emb[:, 2],
            "u_emb_3": user_emb[:, 3],
        }
    )

    # 2. 视频侧特征（对齐 video_features_basic_pure.csv 核心字段）
    video_emb = rng.normal(0, 1, size=(n_videos, latent_dim))
    # 制造强长尾流行度（Zipf / 指数偏斜）
    pop_score = rng.normal(0, 1.35, size=n_videos)
    # 真实内容质量：与流行度仅弱相关，确保存在大量“冷门但优质”与“热门但一般”的视频
    true_quality = 0.25 * pop_score + rng.normal(0, 1.0, size=n_videos)
    duration_ms = np.clip(rng.lognormal(mean=np.log(25_000), sigma=0.7, size=n_videos), 4_000, 180_000).astype(int)
    video_type = rng.choice(["NORMAL", "AD"], size=n_videos, p=[0.94, 0.06])
    tag = rng.integers(0, 30, size=n_videos)

    video_features = pd.DataFrame(
        {
            "video_id": np.arange(n_videos),
            "author_id": rng.integers(0, 600, size=n_videos),
            "video_type": video_type,
            "video_duration": duration_ms,
            "tag": tag,
            "v_emb_0": video_emb[:, 0],
            "v_emb_1": video_emb[:, 1],
            "v_emb_2": video_emb[:, 2],
            "v_emb_3": video_emb[:, 3],
            # 下面两列仅用于展示真实机制/生成数据，不作为模型作弊特征
            "_true_pop_logit": pop_score,
            "_true_quality": true_quality,
        }
    )

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _simulate_feedback(u_ids: np.ndarray, v_ids: np.ndarray, is_rand: int) -> pd.DataFrame:
        match = np.sum(user_emb[u_ids] * video_emb[v_ids], axis=1) / np.sqrt(latent_dim)
        ad_penalty = np.where(video_type[v_ids] == "AD", -0.7, 0.0)
        # 真实点击概率：取决于匹配度 + 真实质量 + 用户活跃度
        click_logit = -1.65 + 0.90 * match + 0.75 * true_quality[v_ids] + active_effect[u_ids] + ad_penalty
        p_click = _sigmoid(click_logit)
        is_click = rng.binomial(1, p_click)

        # 长观、点赞、关注、评论、转发、负反馈、播放时长
        p_long = _sigmoid(-0.8 + 0.7 * match + 0.6 * true_quality[v_ids]) * is_click
        long_view = rng.binomial(1, np.clip(p_long, 0, 1))
        is_like = rng.binomial(1, np.clip(_sigmoid(-2.4 + 0.8 * match + 0.7 * true_quality[v_ids]) * is_click, 0, 1))
        is_follow = rng.binomial(1, np.clip(_sigmoid(-3.8 + 0.6 * match + 0.6 * true_quality[v_ids]) * is_click, 0, 1))
        is_comment = rng.binomial(1, np.clip(_sigmoid(-4.1 + 0.5 * true_quality[v_ids]) * is_click, 0, 1))
        is_forward = rng.binomial(1, np.clip(_sigmoid(-4.4 + 0.5 * true_quality[v_ids]) * is_click, 0, 1))
        # 不喜欢的视频更容易触发 hate
        p_hate = _sigmoid(-4.2 - 0.6 * match - 0.5 * true_quality[v_ids])
        is_hate = rng.binomial(1, np.clip(p_hate, 0, 1))

        dur = duration_ms[v_ids]
        play_ratio = np.where(
            is_click == 1,
            np.clip(rng.beta(2.5 + 2.0 * long_view, 2.0, size=len(u_ids)) * 1.15, 0.05, 1.5),
            np.clip(rng.beta(0.6, 5.0, size=len(u_ids)), 0.0, 0.25),
        )
        play_time_ms = (dur * play_ratio).astype(int)

        if is_rand == 1:
            tab = rng.choice([1, 2], size=len(u_ids), p=[0.75, 0.25])
        else:
            tab = rng.choice([0, 1, 2, 4, 6], size=len(u_ids), p=[0.18, 0.48, 0.16, 0.10, 0.08])

        return pd.DataFrame(
            {
                "user_id": u_ids,
                "video_id": v_ids,
                "date": rng.integers(20220422, 20220509, size=len(u_ids)),
                "is_click": is_click,
                "long_view": long_view,
                "is_like": is_like,
                "is_follow": is_follow,
                "is_comment": is_comment,
                "is_forward": is_forward,
                "is_hate": is_hate,
                "play_time_ms": play_time_ms,
                "duration_ms": dur,
                "is_rand": is_rand,
                "tab": tab,
            }
        )

    # 3. 常规有偏曝光 log_standard：
    # 高活用户获得更多曝光 + 线上策略强偏好高流行度与高兴趣匹配视频
    user_std_prob = np.array([0.08, 0.22, 0.35, 0.35])[active_code]
    user_std_prob = user_std_prob / user_std_prob.sum()
    u_std = rng.choice(n_users, size=n_std_logs, p=user_std_prob)

    # 为每个曝光采样候选集并按“强流行度偏置 + 精排匹配分”选出曝光视频（模拟真实召回+精排选择偏差）
    cand_k = 12
    cand_videos = rng.integers(0, n_videos, size=(n_std_logs, cand_k))
    cand_match = np.einsum("ij,ikj->ik", user_emb[u_std], video_emb[cand_videos]) / np.sqrt(latent_dim)
    # 线上精排打分里混入了很强的流行度偏差 (1.55 * pop_score) 与匹配度 (1.15 * match)
    exposure_logits = 1.55 * pop_score[cand_videos] + 1.15 * cand_match + 0.30 * true_quality[cand_videos]
    exposure_logits -= exposure_logits.max(axis=1, keepdims=True)
    exp_probs = np.exp(exposure_logits)
    exp_probs /= exp_probs.sum(axis=1, keepdims=True)
    # 按概率采样被曝光的视频
    cdf = np.cumsum(exp_probs, axis=1)
    r = rng.random(size=(n_std_logs, 1))
    chosen_idx = (cdf < r).sum(axis=1)
    chosen_idx = np.clip(chosen_idx, 0, cand_k - 1)
    v_std = cand_videos[np.arange(n_std_logs), chosen_idx]

    log_standard = _simulate_feedback(u_std, v_std, is_rand=0)

    # 4. 随机无偏曝光 log_random：用户覆盖更均匀，视频完全均匀随机抽取（MAR）
    u_rand = rng.integers(0, n_users, size=n_rand_logs)
    v_rand = rng.integers(0, n_videos, size=n_rand_logs)
    log_random = _simulate_feedback(u_rand, v_rand, is_rand=1)

    return KuaiRandBundle(
        log_standard=log_standard,
        log_random=log_random,
        user_features=user_features,
        video_features=video_features,
        is_synthetic=True,
        source_desc="SYNTHETIC KuaiRand-Pure Schema Fallback (seed=42)",
    )


def load_kuairand_pure(
    max_std_rows: Optional[int] = 300_000,
    max_rand_rows: Optional[int] = 300_000,
) -> KuaiRandBundle:
    """统一入口：有真实数据则读真实数据，否则自动回退到对齐 Schema 的仿真数据。"""
    real_dir = _find_real_data_dir()
    if real_dir is not None:
        return _load_real_kuairand_pure(real_dir, max_std_rows=max_std_rows, max_rand_rows=max_rand_rows)
    return _generate_synthetic_kuairand_pure()
