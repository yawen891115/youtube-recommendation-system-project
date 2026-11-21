"""Reproducible chronological training: like classifier, pre-watch ranker, reports, artifacts.

Data: 100,000 real Kuaishou plays from KuaiRand-Pure (see data/README.md).
1. Post-watch like classifier (original benchmark): LR, RF, GB, LightGBM + Optuna, AUC.
2. Pre-watch video ranker: candidate lists, NDCG@10 benchmark, LightGBM LambdaRank + Optuna,
   paired bootstrap significance test against the best baseline.
Both use the same chronological 64K / 16K / 20K split and the same point-in-time features.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data_validation import chronological_split, describe_data, load_data
from .feature_engineering import FeatureEngineer
from .train_ranker import EXPERIMENT as RANKING_EXPERIMENT
from .train_ranker import FINAL_MODEL, log_ranking_runs, run_ranking_benchmark

logger = logging.getLogger(__name__)
CLASSIFIER_EXPERIMENT = "youtube-shorts-chronological"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def evaluate_model(model, X, y):
    probabilities = np.clip(model.predict_proba(X)[:, 1], 0, 1)
    predictions = probabilities >= 0.5
    return {
        "auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probabilities)),
    }


def fit_lightgbm(model, X_train, y_train, X_validation, y_validation):
    return model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(25, first_metric_only=True, verbose=False)],
    )


def data_diagnostics(dataset, splits):
    """What structure can a pre-watch ranker exploit? Training split unless noted."""
    train, test = splits["train"], splits["test"]
    plays = train.video_id.value_counts()
    top = max(1, int(len(plays) * 0.01))
    half = len(train) // 2
    early, late = (
        train.iloc[:half].video_id.value_counts(),
        train.iloc[half:].video_id.value_counts(),
    )
    both = pd.concat([early, late], axis=1).fillna(0)
    creators = train.merge(dataset.videos[["video_id", "creator_id", "category"]], on="video_id")
    repeat_creator = creators.duplicated(["user_id", "creator_id"]).mean()
    per_user = creators.groupby("user_id").category.agg(lambda s: s.value_counts().iloc[0] / len(s))
    active = creators.groupby("user_id").size() >= 5
    return {
        "top_1pct_videos_share_of_plays": float(plays.iloc[:top].sum() / plays.sum()),
        "popularity_rank_correlation_first_vs_second_half": float(
            both.corr(method="spearman").iloc[0, 1]
        ),
        "plays_repeating_a_creator_already_played": float(repeat_creator),
        "mean_top_category_share_users_with_5_plays": float(per_user[active].mean()),
        "test_users_seen_in_training": float(test.user_id.isin(train.user_id).mean()),
        "test_videos_seen_in_training": float(test.video_id.isin(train.video_id).mean()),
    }


def run_training(
    data_dir="data",
    model_dir="models",
    report_dir="reports",
    trials=5,
    ranker_trials=12,
    train_negatives=10,
    seed=42,
    sample_size=None,
    tracking_uri=None,
):
    if trials < 1 or ranker_trials < 1:
        raise ValueError("At least one optimization trial is required")
    root, reports = Path(model_dir), Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    dataset = load_data(data_dir)
    statistics = describe_data(dataset)
    frame = dataset.interactions
    if sample_size is not None:
        frame = frame.iloc[:sample_size].copy()
    splits = chronological_split(frame)
    statistics["signal"] = data_diagnostics(dataset, splits)
    logger.info("Fitting point-in-time features on %d training events", len(splits["train"]))
    # Point-in-time features read the whole play log; learned inputs use training plays only.
    engineer = FeatureEngineer(seed=seed).fit(
        splits["train"], dataset.users, dataset.videos, history=frame
    )
    features = {name: engineer.transform(part) for name, part in splits.items()}
    labels = {name: part.liked.to_numpy(int) for name, part in splits.items()}
    engineer.save_artifacts(root)
    processed = []
    for name, part in splits.items():
        meta = part[["user_id", "video_id", "timestamp", "liked", "shared"]].reset_index(drop=True)
        meta["split"] = name
        processed.append(pd.concat([meta, features[name]], axis=1))
    combined = pd.concat(processed, ignore_index=True)
    combined.to_parquet(
        Path(data_dir) / "processed_interactions.parquet", index=False, compression="zstd"
    )
    # Compressed CSV provides the assignment's interchange format without a >100 MB Git blob.
    combined.to_csv(
        Path(data_dir) / "processed_interactions.csv.xz",
        index=False,
        compression={"method": "xz", "preset": 3},
    )
    X, V, T = (features[name] for name in ["train", "validation", "test"])
    y, vy, ty = (labels[name] for name in ["train", "validation", "test"])
    Path("experiments").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri or "sqlite:///experiments/mlflow.db")
    mlflow.set_experiment(CLASSIFIER_EXPERIMENT)

    # ---------------------------------------------- part 1: post-watch like classifier
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=400, solver="lbfgs", random_state=seed
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=2,
            random_state=seed,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=80, max_depth=2, subsample=0.6, max_features="sqrt", random_state=seed
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            is_unbalance=True,
            n_jobs=2,
            verbosity=-1,
            random_state=seed,
        ),
    }
    comparisons, runs = {}, []
    for name, model in models.items():
        start = time.perf_counter()
        logger.info("Training %s", name)
        if name == "lightgbm":
            fit_lightgbm(model, X, y, V, vy)
        else:
            model.fit(X, y)
        metrics = evaluate_model(model, V, vy)
        metrics["training_seconds"] = time.perf_counter() - start
        comparisons[name] = metrics
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(
                {
                    "model": name,
                    "seed": seed,
                    "train_samples": len(X),
                    "validation_samples": len(V),
                    "feature_count": X.shape[1],
                }
            )
            mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
            runs.append({"run_id": run.info.run_id, "model": name, "validation": metrics})

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        }
        model = lgb.LGBMClassifier(
            **params, n_estimators=400, is_unbalance=True, n_jobs=2, random_state=seed, verbosity=-1
        )
        fit_lightgbm(model, X, y, V, vy)
        trial.set_user_attr("best_iteration", int(model.best_iteration_))
        return roc_auc_score(vy, model.predict_proba(V)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    tuned = lgb.LGBMClassifier(
        **study.best_params,
        n_estimators=400,
        is_unbalance=True,
        n_jobs=2,
        random_state=seed,
        verbosity=-1,
    )
    fit_lightgbm(tuned, X, y, V, vy)
    models["lightgbm_optuna"] = tuned
    comparisons["lightgbm_optuna"] = evaluate_model(tuned, V, vy)
    with mlflow.start_run(run_name="lightgbm_optuna") as run:
        mlflow.log_params(study.best_params)
        mlflow.log_metrics(
            {f"validation_{key}": value for key, value in comparisons["lightgbm_optuna"].items()}
        )
        runs.append(
            {
                "run_id": run.info.run_id,
                "model": "lightgbm_optuna",
                "validation": comparisons["lightgbm_optuna"],
                "params": study.best_params,
            }
        )
    best_name = max(comparisons, key=lambda name: comparisons[name]["auc"])
    best = models[best_name]
    test_metrics = evaluate_model(best, T, ty)

    # ------------------------------------------------ part 2: pre-watch video ranker
    ranker, ranking_columns, ranking = run_ranking_benchmark(
        dataset,
        splits,
        engineer,
        trials=ranker_trials,
        train_negatives=train_negatives,
        seed=seed,
        frozen_engineer=FeatureEngineer(seed=seed).fit(
            splits["train"], dataset.users, dataset.videos
        ),
    )
    with mlflow.start_run(run_name="held_out_evaluation") as run:
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        runs.append({"run_id": run.info.run_id, "model": best_name, "test": test_metrics})
    runs += log_ranking_runs(ranking)
    mlflow.set_experiment(CLASSIFIER_EXPERIMENT)

    model_file = root / f"best_model_{best_name}.pkl"
    joblib.dump(best, model_file, compress=3)
    joblib.dump(ranker, root / "ranking_model.pkl", compress=3)
    weights = (
        best.named_steps["logisticregression"].coef_[0]
        if hasattr(best, "named_steps")
        else best.feature_importances_
    )
    importance = sorted(
        [
            {"feature": feature, "importance": float(abs(weight))}
            for feature, weight in zip(X.columns, weights)
        ],
        key=lambda row: -row["importance"],
    )
    final = ranking["models"][FINAL_MODEL]
    benchmark_table = [
        {
            "model": name,
            "family": entry["family"],
            "description": entry["description"],
            "validation_ndcg@10": entry["validation"]["ndcg@10"],
            "test_ndcg@10": entry["test"]["ndcg@10"],
            "test_ndcg@10_low": entry["test_ci"]["ndcg@10"]["low"],
            "test_ndcg@10_high": entry["test_ci"]["ndcg@10"]["high"],
            "test_hr@10": entry["test"]["hr@10"],
        }
        for name, entry in ranking["models"].items()
    ]
    metadata = {
        "model_version": hashlib.sha256(
            model_file.read_bytes()
            + (root / "ranking_model.pkl").read_bytes()
            + (root / "feature_engineer.joblib").read_bytes()
        ).hexdigest()[:12],
        "model_file": model_file.name,
        "model_name": best_name,
        "feature_columns": list(X.columns),
        "ranking_feature_columns": ranking_columns,
        "ranking_model": {
            "name": FINAL_MODEL,
            "type": "lightgbm.LGBMRanker (objective=lambdarank)",
            "params": final["params"],
            "test_metrics": final["test"],
            "test_ci": final["test_ci"],
            "validation_metrics": final["validation"],
            "best_baseline": ranking["best_baseline"],
            "best_baseline_test_metrics": ranking["models"][ranking["best_baseline"]]["test"],
            "significance_vs_best_baseline": ranking["significance_vs_best_baseline"],
            "protocol": ranking["protocol"],
        },
        "ranking_benchmark": benchmark_table,
        "ranking_ablations": ranking["ablations"],
        "ranking_feature_importance": ranking["feature_importance"][:40],
        "ranking_feature_group_importance": ranking["feature_group_importance"],
        "training_cutoff": engineer.training_cutoff.isoformat(),
        "embedding_cutoff": engineer.embedding_cutoff.isoformat(),
        "feature_importance": importance,
        "validation_metrics": comparisons[best_name],
        "test_metrics": test_metrics,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib"]
        },
        "artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.pkl")
        },
    }
    metadata["artifact_sha256"]["feature_engineer.joblib"] = hashlib.sha256(
        (root / "feature_engineer.joblib").read_bytes()
    ).hexdigest()
    write_json(root / "metadata.json", metadata)
    split_info = {
        name: {
            "rows": len(part),
            "positive_rate": float(part.liked.mean()),
            "start": part.timestamp.min().isoformat(),
            "end": part.timestamp.max().isoformat(),
        }
        for name, part in splits.items()
    }
    result = {
        "seed": seed,
        "data": statistics,
        "split": split_info,
        "features": X.shape[1],
        "validation_models": comparisons,
        "selected_model": best_name,
        "test": test_metrics,
        "ranking_test": {
            "model": FINAL_MODEL,
            "ndcg@10": final["test"]["ndcg@10"],
            "hr@10": final["test"]["hr@10"],
            "best_baseline": ranking["best_baseline"],
            "best_baseline_ndcg@10": ranking["models"][ranking["best_baseline"]]["test"]["ndcg@10"],
            "paired_bootstrap_p_value": ranking["significance_vs_best_baseline"]["p_value"],
        },
        "optuna": {
            "trials": trials,
            "best_params": study.best_params,
            "best_validation_auc": study.best_value,
        },
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(data_dir).glob("*.csv")
            if p.name != "processed_interactions.csv"
        },
    }
    write_json(reports / "training_results.json", result)
    write_json(reports / "ranking_results.json", ranking)
    write_json(reports / "mlflow_runs.json", runs)
    study.trials_dataframe().to_csv(reports / "optuna_trials.csv", index=False)
    pd.DataFrame(ranking["optuna"]["trial_log"]).to_csv(
        reports / "ranker_optuna_trials.csv", index=False
    )
    generate_reports(result, reports)
    generate_ranking_report(ranking, statistics["signal"], reports)
    return result, ranking


# ------------------------------------------------------------------------------ reports
SURFACE, INK, INK_2, MUTED, GRID, ACCENT, NEUTRAL = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#2a78d6",
    "#c3c2b7",
)


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(NEUTRAL)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def generate_reports(result, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["validation_models"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=SURFACE)
    values = [result["validation_models"][name]["auc"] for name in names]
    axes[0].barh(names, values, color=ACCENT, height=0.4)
    axes[0].axvline(0.5, color=MUTED, linewidth=1)
    axes[0].text(0.505, len(names) - 0.4, "chance", color=MUTED, fontsize=8)
    for index, value in enumerate(values):
        axes[0].text(value + 0.01, index, f"{value:.3f}", va="center", color=INK, fontsize=8)
    axes[0].set(xlim=(0, 1), xlabel="Validation ROC AUC")
    axes[0].set_title("Like classifier (post-watch), validation", loc="left", color=INK)
    counts = result["data"]["category_counts"]
    axes[1].barh(list(counts)[::-1], list(counts.values())[::-1], color=ACCENT, height=0.4)
    axes[1].set(xlabel="Catalog videos")
    axes[1].set_title("Ten largest video categories (primary tag)", loc="left", color=INK)
    for ax in axes:
        _axes_style(ax)
    fig.tight_layout()
    fig.savefig(directory / "model-and-data-overview.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    lines = [
        "# Experiment report: post-watch like classifier",
        "",
        "Results below come from 100,000 real KuaiRand-Pure plays with chronological holdouts. "
        "The pre-watch video ranker has its own report: [ranking-report.md](ranking-report.md).",
        "",
        "| Model | Validation AUC | Accuracy | Average precision |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metrics in result["validation_models"].items():
        lines.append(
            f"| {name} | {metrics['auc']:.4f} | {metrics['accuracy']:.4f} | {metrics['average_precision']:.4f} |"
        )
    ranking = result["ranking_test"]
    lines += [
        "",
        f"Selected by validation AUC: **{result['selected_model']}**.",
        "",
        f"Held-out test AUC: **{result['test']['auc']:.4f}**; average precision: **{result['test']['average_precision']:.4f}** (like rate {result['split']['test']['positive_rate']:.2%}).",
        f"Pre-watch ranker ({ranking['model']}): test NDCG@10 **{ranking['ndcg@10']:.4f}** vs {ranking['best_baseline_ndcg@10']:.4f} for the best baseline ({ranking['best_baseline']}); paired bootstrap p = {ranking['paired_bootstrap_p_value']:.4g}.",
        "",
        "Probabilities are uncalibrated model scores. Interpret accuracy alongside AUC and average precision: likes are rare, so the majority-only accuracy baseline is high. Offline metrics do not establish online recommendation quality; no online CTR or watch-time uplift was measured.",
        "",
        "![Model comparison and catalog distribution](model-and-data-overview.png)",
        "",
        "## Protocol",
        "",
        "Plays are ordered by timestamp and split 64,000 / 16,000 / 20,000. Equal timestamps remain together. Label prevalence is reported for every split. Class weighting is applied within training rather than shuffling future records into the past.",
        "",
        "History aggregates use plays strictly before each prediction timestamp and are frozen at the training cutoff for validation and test. SVD uses only the first 10% of training history; warmup rows receive zero embeddings. Training and serving reuse the same serialized feature pipeline.",
        "",
        "Completion-rate features describe already-observed watch behavior for the like-prediction API. The pre-watch ranker excludes those features.",
        "",
        "Optuna and early stopping use validation only. The final model is selected before the held-out test is evaluated. The random seed is 42.",
        "",
        "## Reproduction",
        "",
        "Run `python -m src.train`. Exact dependency versions are recorded in `requirements-lock.txt`. See `training_results.json`, `optuna_trials.csv`, and `mlflow_runs.json` for the full measurements.",
        "",
    ]
    (directory / "experiment-report.md").write_text("\n".join(lines))
    data = result["data"]
    signal = data["signal"]
    (directory / "data-exploration.md").write_text(
        "# Data exploration\n\n"
        f"The dataset holds {data['interactions']:,} real plays by {data['users']:,} users over a "
        f"catalog of {data['videos']:,} short videos ({data['videos_played']:,} of them played; "
        f"{data['categories']} primary-tag categories), from {data['first_interaction']} to "
        f"{data['last_interaction']}. Source: KuaiRand-Pure (Kuaishou), CC BY-SA 4.0; the "
        "subset rules are in `data/README.md`.\n\n"
        f"Like rate: {data['positive_rate']:.2%}. Share rate: {data['share_rate']:.2%}. "
        f"Long-view rate: {data['long_view_rate']:.2%}. Mean watch duration: "
        f"{data['mean_watch_seconds']:.1f} seconds. Plays longer than the video (rewatch "
        f"loops): {data['rewatch_rows']:,}. Plays dated before upload: "
        f"{data['interactions_before_upload']:,}.\n\n"
        "## Structure a pre-watch ranker can use (training split)\n\n"
        "| Check | Value |\n| --- | ---: |\n"
        f"| Share of plays on the top 1% of videos | {signal['top_1pct_videos_share_of_plays']:.1%} |\n"
        f"| Video popularity rank correlation, first vs second half of training | {signal['popularity_rank_correlation_first_vs_second_half']:.3f} |\n"
        f"| Plays of a creator the user had already played | {signal['plays_repeating_a_creator_already_played']:.1%} |\n"
        f"| Mean share of a user's plays in their top category (users with 5+ plays) | {signal['mean_top_category_share_users_with_5_plays']:.1%} |\n"
        f"| Test plays by users seen in training | {signal['test_users_seen_in_training']:.1%} |\n"
        f"| Test plays of videos seen in training | {signal['test_videos_seen_in_training']:.1%} |\n\n"
        "## Split distributions\n\n"
        + "\n".join(
            f"- {name}: {part['rows']:,} plays, {part['positive_rate']:.2%} liked, "
            f"{part['start']} to {part['end']}."
            for name, part in result["split"].items()
        )
        + "\n"
    )


def generate_ranking_report(ranking, signal, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    models = ranking["models"]
    final, baseline = ranking["final_model"], ranking["best_baseline"]
    order = sorted(models, key=lambda name: models[name]["test"]["ndcg@10"])
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    values = [models[name]["test"]["ndcg@10"] for name in order]
    low = [values[i] - models[name]["test_ci"]["ndcg@10"]["low"] for i, name in enumerate(order)]
    high = [models[name]["test_ci"]["ndcg@10"]["high"] - values[i] for i, name in enumerate(order)]
    colors = [ACCENT if name == final else NEUTRAL for name in order]
    labels = [f"{name} (best baseline)" if name == baseline else name for name in order]
    ax.barh(labels, values, color=colors, height=0.36)
    ax.errorbar(values, labels, xerr=[low, high], fmt="none", ecolor=INK_2, elinewidth=1, capsize=0)
    for index, value in enumerate(values):
        ax.text(
            value + high[index] + 0.004, index, f"{value:.3f}", va="center", color=INK, fontsize=8
        )
    ax.set_xlabel("Test NDCG@10 (1 played + 99 sampled videos per query; 95% bootstrap CI)")
    ax.set_title("Pre-watch ranking benchmark, chronological test split", loc="left", color=INK)
    ax.set_xlim(0, max(v + h for v, h in zip(values, high)) * 1.15)
    _axes_style(ax)
    fig.tight_layout()
    fig.savefig(directory / "ranking-benchmark.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    top = ranking["feature_importance"][:15][::-1]
    total = sum(row["gain"] for row in ranking["feature_importance"]) or 1.0
    shares = [row["gain"] / total for row in top]
    axes[0].barh([row["feature"] for row in top], shares, color=ACCENT, height=0.5)
    for index, value in enumerate(shares):
        axes[0].text(value + 0.003, index, f"{value:.1%}", va="center", color=INK, fontsize=8)
    axes[0].set_title("Top 15 features, share of total gain", loc="left", color=INK)
    groups = ranking["feature_group_importance"][::-1]
    axes[1].barh(
        [row["group"] for row in groups], [row["share"] for row in groups], color=ACCENT, height=0.3
    )
    for index, row in enumerate(groups):
        axes[1].text(
            row["share"] + 0.005, index, f"{row['share']:.1%}", va="center", color=INK, fontsize=8
        )
    axes[1].set_title("Gain share by feature family", loc="left", color=INK)
    for ax in axes:
        _axes_style(ax)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    fig.savefig(directory / "feature-importance.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    protocol = ranking["protocol"]
    test_gain = ranking["significance_vs_best_baseline"]

    def interval(row):
        return (
            f"{row['mean_difference']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}], "
            f"p = {row['p_value']:.4g}"
        )

    lines = [
        "# Ranking report: pre-watch video ranker",
        "",
        "## Protocol",
        "",
        f"- Split: {protocol['split']}.",
        f"- Query: {protocol['query']}; {protocol['candidates_per_query']} candidates per query "
        f"(the played video + {protocol['candidates_per_query'] - 1} sampled videos).",
        f"- Negatives: {protocol['negatives']}. Training lists use "
        f"{protocol['train_negatives_per_query']} negatives per query.",
        f"- Features: {protocol['features']}; {ranking['ranking_features']} ranking features.",
        f"- Relevance: {protocol['relevance']}. Metric: {protocol['metric']}.",
        f"- Model selection: {protocol['model_selection']} (validation uses "
        f"{protocol['validation_queries_for_selection']:,} randomly chosen validation queries).",
        f"- Uncertainty: {protocol['confidence_intervals']}; {protocol['significance']}.",
        f"- Queries: train {ranking['queries']['train']:,}, validation "
        f"{ranking['queries']['validation']:,}, test {ranking['queries']['test']:,} "
        f"(all first-time plays in the test split).",
        "",
        "## Results (test split, scored once)",
        "",
        "| Model | Family | Validation NDCG@10 | Test NDCG@10 [95% CI] | Test HR@10 | Test MRR | Test AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(models, key=lambda n: -models[n]["test"]["ndcg@10"]):
        entry = models[name]
        ci = entry["test_ci"]["ndcg@10"]
        label = (
            f"**{name}**"
            if name == final
            else (f"{name} (best baseline)" if name == baseline else name)
        )
        lines.append(
            f"| {label} | {entry['family']} | {entry['validation']['ndcg@10']:.4f} | "
            f"{entry['test']['ndcg@10']:.4f} [{ci['low']:.4f}, {ci['high']:.4f}] | "
            f"{entry['test']['hr@10']:.4f} | {entry['test']['mrr']:.4f} | {entry['test']['auc']:.4f} |"
        )
    lines += [
        "",
        f"Best baseline, chosen on validation NDCG@10 before testing: **{baseline}**.",
        "",
        f"**{final} minus {baseline}, test NDCG@10 (paired bootstrap over "
        f"{test_gain['queries']:,} queries, {test_gain['resamples']:,} resamples):** "
        f"{interval(test_gain)}.",
        "",
        "![Ranking benchmark](ranking-benchmark.png)",
        "",
        "## Ablations (same tuned parameters, one change each)",
        "",
        "| Variant | Features removed | Validation NDCG@10 | Test NDCG@10 | Final minus variant (paired) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in ranking["ablations"].items():
        lines.append(
            f"| {name} | {row['removed_features']} | {row['validation_ndcg@10']:.4f} | "
            f"{row['test_ndcg@10']:.4f} | {interval(row['difference_vs_final'])} |"
        )
    frozen = ranking.get("frozen_history_sensitivity") or {}
    if frozen:
        gap = frozen["final_vs_strongest_frozen_baseline"]
        rival = frozen["strongest_baseline_on_frozen_test"]
        lines += [
            "",
            "## Sensitivity: histories frozen at the training cutoff",
            "",
            "Same trained models, but test features read training plays only, as a service "
            "without online feature updates would. Time-relative features (recency, 24-hour "
            "plays) then drift outside their training range over the 17-day test window.",
            "",
            "| Model | Test NDCG@10, full-log features | Test NDCG@10, frozen features |",
            "| --- | ---: | ---: |",
            *[
                f"| {name} | {models[name]['test']['ndcg@10']:.4f} | {value:.4f} |"
                for name, value in sorted(frozen["test_ndcg@10"].items(), key=lambda item: -item[1])
            ],
            "",
            f"With frozen features, {final} minus the strongest baseline under the same "
            f"features ({rival}): {interval(gap)}.",
        ]
    lines += [
        "",
        "## Feature importance (LightGBM gain)",
        "",
        "![Feature importance](feature-importance.png)",
        "",
        "| Feature family | Share of gain |",
        "| --- | ---: |",
        *[
            f"| {row['group']} | {row['share']:.1%} |"
            for row in ranking["feature_group_importance"]
        ],
        "",
        "Gain counts every split, including splits on noisy features, so a family can collect "
        "gain without improving NDCG; the ablation table is the causal check.",
        "",
        "## Reading the results",
        "",
        f"In training, the top 1% of videos receive {signal['top_1pct_videos_share_of_plays']:.1%} "
        f"of plays and {signal['plays_repeating_a_creator_already_played']:.1%} of plays repeat a "
        "creator the user had already played, so popularity, freshness and per-user watch history "
        "all carry signal. Sampled metrics (1 + 99 candidates) rank the true play among random "
        "catalog videos; they are not full-catalog metrics and do not establish online "
        "engagement lift.",
        "",
        f"Optuna: {ranking['optuna']['trials']} trials, best validation NDCG@10 "
        f"{ranking['optuna']['best_validation_ndcg@10']:.4f}. Full numbers: "
        "`ranking_results.json`, `ranker_optuna_trials.csv`, MLflow experiment "
        f"`{RANKING_EXPERIMENT}`.",
        "",
    ]
    (directory / "ranking-report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--trials", type=int, default=5, help="Optuna trials, like classifier")
    parser.add_argument("--ranker-trials", type=int, default=12, help="Optuna trials, ranker")
    parser.add_argument("--train-negatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--tracking-uri")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # lightgbm>=4.6 renamed eval_set; the old name keeps compatibility with lightgbm 4.3-4.5.
    warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")
    with threadpool_limits(limits=2):
        result, ranking = run_training(**vars(args))
    final = ranking["final_model"]
    print(
        json.dumps(
            {
                "selected_classifier": result["selected_model"],
                "classifier_test_auc": result["test"]["auc"],
                "ranker": final,
                "ranker_test": ranking["models"][final]["test"],
                "best_baseline": ranking["best_baseline"],
                "best_baseline_test": ranking["models"][ranking["best_baseline"]]["test"],
                "paired_bootstrap": ranking["significance_vs_best_baseline"],
            },
            indent=2,
        )
    )
