"""Reproducible chronological model comparison, optimization, and artifact export."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import time
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
from .feature_engineering import OBSERVED_FEATURES, FeatureEngineer

logger = logging.getLogger(__name__)


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


def run_training(
    data_dir="data",
    model_dir="models",
    report_dir="reports",
    trials=5,
    seed=42,
    sample_size=None,
    tracking_uri=None,
):
    if trials < 1:
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
    logger.info("Fitting point-in-time features on %d training events", len(splits["train"]))
    engineer = FeatureEngineer(seed=seed).fit(splits["train"], dataset.users, dataset.videos)
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
    mlflow.set_experiment("youtube-shorts-chronological")
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
    # The recommendation model cannot use watch duration observed after serving a video.
    ranking_columns = [column for column in X.columns if column not in OBSERVED_FEATURES]
    ranker = lgb.LGBMClassifier(
        **study.best_params,
        n_estimators=400,
        is_unbalance=True,
        n_jobs=2,
        random_state=seed,
        verbosity=-1,
    )
    fit_lightgbm(ranker, X[ranking_columns], y, V[ranking_columns], vy)
    rank_test = evaluate_model(ranker, T[ranking_columns], ty)
    with mlflow.start_run(run_name="held_out_evaluation") as run:
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metrics({f"ranking_test_{key}": value for key, value in rank_test.items()})
        runs.append(
            {
                "run_id": run.info.run_id,
                "model": best_name,
                "test": test_metrics,
                "ranking_test": rank_test,
            }
        )
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
        "ranking_test": rank_test,
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
    write_json(reports / "mlflow_runs.json", runs)
    study.trials_dataframe().to_csv(reports / "optuna_trials.csv", index=False)
    generate_reports(result, reports)
    return result


def generate_reports(result, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["validation_models"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(
        names, [result["validation_models"][name]["auc"] for name in names], color="#277DA1"
    )
    axes[0].axvline(0.5, color="grey", linestyle="--", linewidth=1)
    axes[0].set(
        xlim=(0, 1), xlabel="Validation ROC AUC", title="Chronological validation comparison"
    )
    axes[1].bar(
        result["data"]["category_counts"].keys(),
        result["data"]["category_counts"].values(),
        color="#43AA8B",
    )
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set(ylabel="Videos", title="Catalog distribution")
    fig.tight_layout()
    fig.savefig(directory / "model-and-data-overview.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Experiment report",
        "",
        "Results below were generated from the bundled dataset with chronological holdouts.",
        "",
        "| Model | Validation AUC | Accuracy | Average precision |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metrics in result["validation_models"].items():
        lines.append(
            f"| {name} | {metrics['auc']:.4f} | {metrics['accuracy']:.4f} | {metrics['average_precision']:.4f} |"
        )
    lines += [
        "",
        f"Selected by validation AUC: **{result['selected_model']}**.",
        "",
        f"Held-out test AUC: **{result['test']['auc']:.4f}**; average precision: **{result['test']['average_precision']:.4f}**.",
        f"Pre-watch ranking model test AUC: **{result['ranking_test']['auc']:.4f}**.",
        "",
        f"Probabilities are uncalibrated model scores. At the default 0.5 decision threshold, test recall is {result['test']['recall']:.4f}; the majority-only accuracy baseline is {1 - result['split']['test']['positive_rate']:.4f}. Interpret accuracy alongside AUC, average precision and recall. Offline discrimination does not establish online recommendation quality. No online CTR, watch-time uplift, or production latency target was measured.",
        "",
        "![Model comparison and catalog distribution](model-and-data-overview.png)",
        "",
        "## Protocol",
        "",
        "Rows are ordered by timestamp and split 64% / 16% / 20%. Equal timestamps remain together. Label prevalence is reported for every split. Class weighting is applied within training rather than shuffling future records into the past.",
        "",
        "History aggregates use events strictly before each prediction timestamp and are frozen at the training cutoff for validation and test. SVD uses only the first 10% of training history; warmup rows receive zero embeddings. Training and serving reuse the same serialized feature pipeline.",
        "",
        "Completion-rate features describe already-observed watch behavior for the like-prediction API. A separate pre-watch recommendation model excludes those features.",
        "",
        "Optuna and early stopping use validation only. The final model is selected before the held-out test is evaluated. The random seed is 42.",
        "",
        "## Reproduction",
        "",
        "Run `python -m src.train --trials 5`. Exact dependency versions are recorded in `requirements-lock.txt`. See `training_results.json`, `optuna_trials.csv`, and `mlflow_runs.json` for the full measurements.",
        "",
    ]
    (directory / "experiment-report.md").write_text("\n".join(lines))
    data = result["data"]
    (directory / "data-exploration.md").write_text(
        f"# Data exploration\n\nThe supplied dataset contains {data['users']:,} users, {data['videos']:,} videos and {data['interactions']:,} interactions.\n\nLike rate: {data['positive_rate']:.2%}. Share rate: {data['share_rate']:.2%}. Mean watch duration: {data['mean_watch_seconds']:.2f} seconds.\n\nObservation window: {data['first_interaction']} to {data['last_interaction']}.\n\nThe three raw tables contain no missing cells or duplicate rows and all interaction foreign keys resolve. Rewatch observations are retained; completion rate is capped at one. There are {data['interactions_before_upload']:,} interactions dated before the catalog upload date. Raw rows are preserved; negative video age is clipped to zero. This inconsistency limits realism. Catalog metadata is treated as a static snapshot; its historical availability cannot be verified. The supplied materials do not establish that this is live YouTube traffic.\n\n## Split distributions\n\n"
        + "\n".join(
            f"- {name}: {part['rows']:,} rows, {part['positive_rate']:.2%} positive."
            for name, part in result["split"].items()
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--tracking-uri")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with threadpool_limits(limits=2):
        result = run_training(**vars(args))
    print(
        json.dumps({"selected_model": result["selected_model"], "test": result["test"]}, indent=2)
    )
