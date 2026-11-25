"""Phase 3 - 实验 3：面向快手「激励效率算法组」的 Uplift / CATE + 预算 ROI 优化实战。

运行方式：
    .venv/bin/python scripts/03_uplift_bridge_demo.py

业务场景模拟：
- 任务：决定是否给用户发放“1 元任务激励红包（T=1 vs T=0）”以提升次日留存（Y ∈ {0,1}）。
- 四象限人群：
  * Sure Things（高活老铁）：自然留存极高，发不发激励增量 ≈ 0（Naive 模型最爱给他们发，纯浪费钱！）
  * Persuadables（中低活价格敏感用户）：发了激励次留大幅提升（真正该投放的目标！）
  * Lost Causes（沉睡流失用户）：发了也不留（增量 ≈ 0）
  * Sleeping Dogs（反感打扰用户）：发了激励反而负向（增量 < 0）
- 训练集：模拟真实业务里的【有偏观测日志 + 不平衡发券（仅约 15% 发券，且历史策略偏爱高活人群）】；
- 测试集：模拟一份干净的【随机对照实验（RCT）测试集】，用于严格计算 Qini / AUUC 与增量留存收益；
- 对比 5 个模型：
  1) Naive 留存预测模型（按 P(Y=1|X) 排序，演示为什么传统推荐思维在激励组会翻车）
  2) S-Learner
  3) T-Learner
  4) X-Learner（专门克制 15% vs 85% 不平衡投放！）
  5) DR-Learner（把实验 2 的 Doubly Robust 思想直接迁到 Uplift 建模！）
- 最后一公里：对比“按增量留存 ΔR 排序” vs “按边际 ROI (ΔR / ΔC) / 拉格朗日对偶决策”在固定预算下的真实收益差距。
"""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def simulate_incentive_data(
    n_samples: int, is_rct: bool, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成带四象限异质性处理效应（HTE）的激励效率仿真数据。

    返回：X, T, Y, true_cate, true_cost_if_treated, segment_name
    """
    rng = np.random.default_rng(seed)

    # 前置协变量 X（全部为干预发生之前可得的特征，杜绝事后穿越！）
    active_days_30d = rng.uniform(0, 30, size=n_samples)          # 过去 30 天活跃天数
    hist_play_min = rng.lognormal(mean=2.8, sigma=0.9, size=n_samples)  # 历史日均观看分钟数
    register_days = rng.uniform(7, 1500, size=n_samples)          # 注册天数
    price_sensitivity = rng.normal(0, 1.0, size=n_samples)        # 价格/激励敏感度（历史领券核销倾向）
    ad_annoyance = rng.normal(0, 1.0, size=n_samples)             # 弹窗反感倾向（历史关闭/负反馈）
    noise_f1 = rng.normal(0, 1.0, size=n_samples)
    noise_f2 = rng.normal(0, 1.0, size=n_samples)

    X = np.column_stack(
        [
            active_days_30d,
            np.log1p(hist_play_min),
            np.log1p(register_days),
            price_sensitivity,
            ad_annoyance,
            noise_f1,
            noise_f2,
        ]
    )

    # 1. 自然基线次留概率 P(Y(0)=1 | X)：强依赖历史活跃度（高活老铁自然次留极高！）
    mu0 = sigmoid(-2.2 + 0.14 * active_days_30d + 0.25 * np.log1p(hist_play_min) + 0.10 * np.log1p(register_days))

    # 2. 真实因果增量 CATE = P(Y(1)=1|X) - P(Y(0)=1|X)：
    #    - 中低活跃度（5~18天）且价格敏感 → Persuadables（高正增量）
    #    - 极高活跃（>24天）→ Sure Things（增量接近 0）
    #    - 极低活跃（<3天）→ Lost Causes（增量很小）
    #    - 高反感用户（ad_annoyance > 1.2）→ Sleeping Dogs（负增量！）
    mid_active_bell = np.exp(-0.5 * ((active_days_30d - 11.0) / 5.5) ** 2)
    raw_lift = (
        0.26 * mid_active_bell * sigmoid(1.2 * price_sensitivity)
        - 0.12 * sigmoid(2.2 * (ad_annoyance - 1.1))
    )
    mu1 = np.clip(mu0 + raw_lift, 0.01, 0.99)
    true_cate = mu1 - mu0

    # 3. 发激励后的预期核销成本 ΔC（单位：元）：
    #    不同用户的激励触达/补贴门槛不同（例如部分低线/轻量任务用户仅需 0.4~0.8 元金币，
    #    而部分高门槛用户需 2.0~3.0 元现金红包），用于清晰对比“纯 ΔR 排序”与“边际 ROI = ΔR/ΔC”的预算效率！
    cost_if_treated = np.clip(
        0.55 + 1.80 * sigmoid(1.6 * noise_f1) + 0.30 * sigmoid(0.10 * (active_days_30d - 12.0)),
        0.40,
        2.80,
    )

    # 4. 干预分配机制 T：
    if is_rct:
        # 干净的 RCT 随机对照实验测试集（50% 发，50% 不发，无混杂）
        e_x = np.full(n_samples, 0.5)
    else:
        # 观测训练集：只有约 ~16% 发激励，且历史策略有严重选择偏差（偏向高活跃人群！）
        e_x = sigmoid(-2.9 + 0.075 * active_days_30d + 0.35 * price_sensitivity)
        e_x = np.clip(e_x, 0.03, 0.70)

    T = rng.binomial(1, e_x)
    Y = np.where(T == 1, rng.binomial(1, mu1), rng.binomial(1, mu0))

    # 仅用于诊断汇报的四象限标签
    seg = np.full(n_samples, "Lost_or_Neutral", dtype=object)
    seg[mu0 >= 0.72] = "Sure_Things"
    seg[true_cate >= 0.12] = "Persuadables"
    seg[true_cate <= -0.04] = "Sleeping_Dogs"

    return X, T, Y, true_cate, cost_if_treated, seg


def compute_qini_auuc(y_true: np.ndarray, t_true: np.ndarray, uplift_score: np.ndarray) -> tuple[float, float]:
    """在 RCT 测试集上计算 Qini 系数（AUUC 归一化面积）与 Top-30% 真实增量次留率。"""
    order = np.argsort(-uplift_score)
    y_ord = y_true[order]
    t_ord = t_true[order]

    cum_t1 = np.cumsum(t_ord == 1)
    cum_t0 = np.cumsum(t_ord == 0)
    cum_y1 = np.cumsum((t_ord == 1) * y_ord)
    cum_y0 = np.cumsum((t_ord == 0) * y_ord)

    # Qini 曲线：cum_y1 - cum_y0 * (cum_t1 / cum_t0)
    ratio = np.divide(cum_t1, cum_t0, out=np.zeros_like(cum_t1, dtype=float), where=cum_t0 > 0)
    qini_curve = cum_y1 - cum_y0 * ratio

    # 随机投放基线（对角线）
    n = len(y_true)
    total_uplift = qini_curve[-1]
    random_curve = np.linspace(0, total_uplift, n)

    # Qini Area = 模型曲线与随机对角线之间的面积（除以总样本数便于阅读）
    qini_area = float(np.mean(qini_curve - random_curve))

    # Top-30% 人群的实际 ATE（实验组均值 - 对照组均值）
    k30 = int(0.30 * n)
    top_y, top_t = y_ord[:k30], t_ord[:k30]
    top30_uplift = float(top_y[top_t == 1].mean() - top_y[top_t == 0].mean())
    return qini_area, top30_uplift


def fit_and_predict_models(
    X_tr: np.ndarray, T_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray
) -> dict[str, np.ndarray]:
    """训练并对比 Naive / S-Learner / T-Learner / X-Learner / DR-Learner。"""
    preds: dict[str, np.ndarray] = {}

    def make_clf() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=120, random_state=42)

    def make_reg() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06, max_iter=120, random_state=42)

    # 1. Naive Retention Model：只预测 P(Y=1 | X)，完全不区分 T
    naive = make_clf().fit(X_tr, Y_tr)
    preds["1_Naive_Retention (反面教材)"] = naive.predict_proba(X_te)[:, 1]

    # 2. S-Learner：把 T 拼进特征训单个模型
    s_model = make_clf().fit(np.column_stack([X_tr, T_tr]), Y_tr)
    s_mu1 = s_model.predict_proba(np.column_stack([X_te, np.ones(len(X_te))]))[:, 1]
    s_mu0 = s_model.predict_proba(np.column_stack([X_te, np.zeros(len(X_te))]))[:, 1]
    preds["2_S-Learner"] = s_mu1 - s_mu0

    # 3. T-Learner：处理组与对照组分别训模型
    m0 = make_clf().fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
    m1 = make_clf().fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
    mu0_te = m0.predict_proba(X_te)[:, 1]
    mu1_te = m1.predict_proba(X_te)[:, 1]
    preds["3_T-Learner"] = mu1_te - mu0_te

    # 估计倾向性得分 e(x) = P(T=1 | X)，用于 X-Learner 权重和 DR-Learner 伪标签
    prop_model = LogisticRegression(C=1.0, max_iter=500).fit(X_tr, T_tr)
    e_tr = np.clip(prop_model.predict_proba(X_tr)[:, 1], 0.03, 0.97)
    e_te = np.clip(prop_model.predict_proba(X_te)[:, 1], 0.03, 0.97)

    # 4. X-Learner：构造反事实伪效应并用 e(x) 加权（克制 15% vs 85% 极度不平衡！）
    mu0_on_t1 = m0.predict_proba(X_tr[T_tr == 1])[:, 1]
    mu1_on_t0 = m1.predict_proba(X_tr[T_tr == 0])[:, 1]
    d1 = Y_tr[T_tr == 1] - mu0_on_t1
    d0 = mu1_on_t0 - Y_tr[T_tr == 0]
    tau1_model = make_reg().fit(X_tr[T_tr == 1], d1)
    tau0_model = make_reg().fit(X_tr[T_tr == 0], d0)
    # 注意权重方向：g(x) = e(x) ≈ 0.16，给高样本基线算出的 tau1 赋 1 - e(x) ≈ 0.84 的高权重！
    preds["4_X-Learner"] = e_te * tau0_model.predict(X_te) + (1.0 - e_te) * tau1_model.predict(X_te)

    # 5. DR-Learner（Doubly Robust 伪标签，与实验 2 的推荐 DR 思想一脉相承！）
    #    用 2-fold cross-fitting 避免过拟合伪标签
    n_tr = len(X_tr)
    psi = np.zeros(n_tr)
    half = n_tr // 2
    folds = [(np.arange(0, half), np.arange(half, n_tr)), (np.arange(half, n_tr), np.arange(0, half))]
    for fit_idx, val_idx in folds:
        f_m0 = make_clf().fit(X_tr[fit_idx][T_tr[fit_idx] == 0], Y_tr[fit_idx][T_tr[fit_idx] == 0])
        f_m1 = make_clf().fit(X_tr[fit_idx][T_tr[fit_idx] == 1], Y_tr[fit_idx][T_tr[fit_idx] == 1])
        f_prop = LogisticRegression(C=1.0, max_iter=500).fit(X_tr[fit_idx], T_tr[fit_idx])

        v_mu0 = f_m0.predict_proba(X_tr[val_idx])[:, 1]
        v_mu1 = f_m1.predict_proba(X_tr[val_idx])[:, 1]
        v_e = np.clip(f_prop.predict_proba(X_tr[val_idx])[:, 1], 0.04, 0.96)
        v_t = T_tr[val_idx]
        v_y = Y_tr[val_idx]

        psi[val_idx] = (
            (v_mu1 - v_mu0)
            + (v_t / v_e) * (v_y - v_mu1)
            - ((1.0 - v_t) / (1.0 - v_e)) * (v_y - v_mu0)
        )
    dr_tau_model = make_reg().fit(X_tr, np.clip(psi, -3.0, 3.0))
    preds["5_DR-Learner (双重稳健)"] = dr_tau_model.predict(X_te)

    return preds


def main() -> None:
    # 生成有偏观测训练集（20,000 人） + 干净 RCT 测试集（15,000 人）
    X_tr, T_tr, Y_tr, cate_tr, _, seg_tr = simulate_incentive_data(20_000, is_rct=False, seed=42)
    X_te, T_te, Y_te, cate_te, cost_te, seg_te = simulate_incentive_data(15_000, is_rct=True, seed=108)

    print("=" * 84)
    print("Phase 3 - 实验 3：激励效率组 Uplift / CATE 建模与预算 ROI 优化演示")
    print("=" * 84)
    print(f"训练集（有偏观测日志）: N={len(X_tr)}, 发激励比例={T_tr.mean()*100:.1f}%")
    print(
        f"  -> 表面直接对比（未纠偏）：发激励组次留={Y_tr[T_tr==1].mean():.3f} vs "
        f"未发组次留={Y_tr[T_tr==0].mean():.3f}，虚假提升=+{Y_tr[T_tr==1].mean()-Y_tr[T_tr==0].mean():.3f}！"
    )
    print(f"  -> 训练集真实平均因果增量（True ATE）：+{cate_tr.mean():.3f}（明显低于表面虚假提升！）")
    print(f"测试集（干净 RCT 流量）: N={len(X_te)}, 发激励比例={T_te.mean()*100:.1f}%, 真实 ATE=+{cate_te.mean():.3f}")

    # 1. 模型对比评估
    preds = fit_and_predict_models(X_tr, T_tr, Y_tr, X_te)
    rows = []
    for name, score in preds.items():
        qini_area, top30_ate = compute_qini_auuc(Y_te, T_te, score)
        # 看看每个模型挑出的 Top-30% 人群里，分别有多少比例是 Persuadables vs Sure Things
        top30_idx = np.argsort(-score)[: int(0.30 * len(X_te))]
        persuadable_share = float((seg_te[top30_idx] == "Persuadables").mean())
        sure_share = float((seg_te[top30_idx] == "Sure_Things").mean())
        cate_rmse = float(np.sqrt(mean_squared_error(cate_te, score))) if "Naive" not in name else np.nan
        rows.append(
            {
                "模型": name,
                "Qini面积(越高越好)": qini_area,
                "Top30%真实增量次留": top30_ate,
                "Top30%含Persuadables": persuadable_share,
                "Top30%含SureThings(浪费)": sure_share,
                "CATE_RMSE(越低越好)": cate_rmse,
            }
        )

    eval_tbl = pd.DataFrame(rows)
    print("\n【1. RCT 测试集 Uplift 模型横向对比】")
    print(eval_tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 2. 最后一公里：固定预算 B 约束下的决策对比（纯 ΔR 排序 vs 边际 ROI = ΔR / ΔC 排序）
    # 用 DR-Learner 的预测增量 ΔR 与已知/估计的增量成本 ΔC 做预算分配
    delta_r_hat = preds["5_DR-Learner (双重稳健)"]
    budget = 2_500.0  # 固定总预算 2,500 元（测试集 15,000 人中仅够覆盖一部分）

    def eval_budget_policy(score_for_rank: np.ndarray, label: str) -> dict[str, float | str]:
        order = np.argsort(-score_for_rank)
        cum_cost = np.cumsum(cost_te[order])
        chosen = order[cum_cost <= budget]
        # 只选预测增量为正的用户
        chosen = chosen[delta_r_hat[chosen] > 0]
        spent = float(cost_te[chosen].sum())
        total_inc_retention = float(cate_te[chosen].sum())
        roi = total_inc_retention / max(spent, 1e-6)
        return {
            "投放策略": label,
            "覆盖人数": len(chosen),
            "消耗预算(元)": spent,
            "真实增量留存人数": total_inc_retention,
            "每千元换回增量留存": roi * 1000.0,
        }

    roi_tbl = pd.DataFrame(
        [
            eval_budget_policy(preds["1_Naive_Retention (反面教材)"], "策略 A：按 Naive 留存分排序（错误）"),
            eval_budget_policy(delta_r_hat, "策略 B：按 CATE (ΔR) 排序（忽略成本差异）"),
            eval_budget_policy(delta_r_hat / cost_te, "策略 C：按边际 ROI (ΔR / ΔC) / 拉格朗日对偶"),
        ]
    )
    print(f"\n【2. 激励效率最后一公里：固定预算 B = {budget:.0f} 元下的真实增量产出对比】")
    print(roi_tbl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n[核心结论]")
    print("  1) Naive 留存模型挑出的 Top-30% 几乎全是 Sure Things（自然高留存用户），真实增量接近 0；")
    print("  2) X-Learner / DR-Learner 在不平衡有偏日志下表现最好，能精准把 Persuadables 捞到最前面；")
    print("  3) 加入成本维度按边际 ROI (ΔR / ΔC) 分配预算，在同样预算下能换回更多真实增量留存！")


if __name__ == "__main__":
    main()
