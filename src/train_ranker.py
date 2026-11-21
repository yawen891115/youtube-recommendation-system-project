"""Pre-watch ranking benchmark and the LightGBM LambdaRank video ranker.

Every model sees the same chronological candidate lists and the same pre-watch features.
Hyperparameters and the best baseline are chosen on validation NDCG@10 only; the test split
is scored once at the end and compared with a paired bootstrap test. See ``src/ranking.py``
for the protocol.
"""

import logging
import time

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import timestamps
from .ranking import (
    N_EVAL_NEGATIVES,
    TOP_K,
    SVDRecommender,
    bootstrap_ci,
    build_candidates,
    category_history_scores,
    feature_group,
    first_plays,
    group_sizes,
    grouped_importance,
    item_knn_scores,
    paired_bootstrap_test,
    per_query_metrics,
    popularity_scores,
    random_scores,
    ranking_columns,
    score_model,
    summarize,
    trending_scores,
)

logger = logging.getLogger(__name__)
EXPERIMENT = "youtube-shorts-ranking"
FINAL_MODEL = "lightgbm_lambdarank"
VALIDATION_QUERIES = 6000
TEST_CHUNK_QUERIES = 2500
BASELINES = {
    "random": ("Reference", "Random order (chance level)"),
    "popularity": ("Heuristic", "Most-played videos in the training window"),
    "trending_24h": ("Heuristic", "Most-played videos in the last 24 hours of training"),
    "category_history": (
        "Content-based",
        "Recency-weighted share of the user's plays in the video's category",
    ),
    "item_knn": ("Collaborative filtering", "Item-item cosine similarity to the user's plays"),
    "svd_mf": ("Collaborative filtering", "Rank-50 truncated SVD of the user x video matrix"),
    "logistic_regression": (
        "Linear",
        "Pointwise logistic regression on the ranker's pre-watch features",
    ),
}
FINAL_DESCRIPTION = (
    "Tree ensemble (learning to rank)",
    "LightGBM LambdaRank on watch-history features and 50-D SVD embeddings",
)
USER_HISTORY_GROUPS = {
    "User watch history",
    "Category & creator watch history",
    "Co-watch history",
}
ABLATIONS = {
    "pointwise_binary_objective": None,
    "without_user_watch_history": lambda c: feature_group(c) in USER_HISTORY_GROUPS,
    "without_svd_embeddings": lambda c: feature_group(c) == "SVD embeddings",
}


def query_frame(engineer, candidates, columns=None):
    frame = engineer.transform_in_chunks(candidates.assign(watch_time=0.0))
    return frame if columns is None else frame[columns]


def select(frame, columns):
    """Column subset without copying when every column is requested."""
    return frame if list(frame.columns) == list(columns) else frame[columns]


def prepare_ranking_data(dataset, splits, engineer, train_negatives, seed):
    """Candidate lists (and features for train / validation) for every split."""
    videos = dataset.videos
    catalog, uploads = videos.video_id.to_numpy(), timestamps(videos.upload_date)
    interactions = dataset.interactions
    interacted = interactions.groupby("user_id").video_id.agg(set).to_dict()
    first = interactions.assign(_first=first_plays(interactions))
    first_keys = set(
        first.loc[first._first, ["user_id", "video_id", "timestamp"]].itertuples(index=False)
    )
    prepared = {}
    for offset, name in enumerate(["train", "validation", "test"]):
        start = time.perf_counter()
        events = splits[name]
        keep = [
            key in first_keys
            for key in events[["user_id", "video_id", "timestamp"]].itertuples(index=False)
        ]
        events = events[np.asarray(keep)]
        total_queries = len(events)
        if name == "validation" and len(events) > VALIDATION_QUERIES:
            chosen = np.random.default_rng(seed).choice(
                len(events), VALIDATION_QUERIES, replace=False
            )
            events = events.iloc[np.sort(chosen)]
        negatives = train_negatives if name == "train" else N_EVAL_NEGATIVES
        candidates = build_candidates(
            events, catalog, uploads, interacted, negatives, seed + offset
        )
        part = {
            "candidates": candidates,
            "labels": candidates.label.to_numpy(),
            "queries": candidates.query_id.to_numpy(),
            "available_queries": int(total_queries),
        }
        if name != "test":
            part["features"] = query_frame(
                engineer, candidates, ranking_columns(engineer.feature_columns)
            )
        prepared[name] = part
        logger.info(
            "%s: %d queries (of %d first plays), %d candidate rows (%.1fs)",
            name,
            candidates.query_id.nunique(),
            total_queries,
            len(candidates),
            time.perf_counter() - start,
        )
    return prepared


def fit_lambdarank(params, data, columns, seed, threads=2, objective="lambdarank"):
    train, validation = data["train"], data["validation"]
    common = dict(n_estimators=2000, random_state=seed, n_jobs=threads, verbosity=-1, **params)
    stop = [lgb.early_stopping(50, first_metric_only=True, verbose=False)]
    if objective == "binary":
        model = lgb.LGBMClassifier(objective="binary", **common)
        model.fit(
            select(train["features"], columns),
            train["labels"],
            eval_set=[(select(validation["features"], columns), validation["labels"])],
            eval_metric="auc",
            callbacks=stop,
        )
        return model
    model = lgb.LGBMRanker(objective="lambdarank", **common)
    model.fit(
        select(train["features"], columns),
        train["labels"],
        group=group_sizes(train["queries"]),
        eval_set=[(select(validation["features"], columns), validation["labels"])],
        eval_group=[group_sizes(validation["queries"])],
        eval_at=[TOP_K],
        callbacks=stop,
    )
    return model


def validation_ndcg(scores, data):
    validation = data["validation"]
    return float(
        per_query_metrics(scores, validation["labels"], validation["queries"])["ndcg@10"].mean()
    )


def run_ranking_benchmark(
    dataset, splits, engineer, trials=12, train_negatives=10, seed=42, frozen_engineer=None
):
    """Train and compare all rankers. Returns (final_model, columns, results).

    ``engineer`` reads the full play log point-in-time (main protocol). ``frozen_engineer``,
    if given, reads training plays only; the same trained models are re-scored on test with
    it as a sensitivity check (histories frozen at the training cutoff).
    """
    start_all = time.perf_counter()
    data = prepare_ranking_data(dataset, splits, engineer, train_negatives, seed)
    columns = ranking_columns(engineer.feature_columns)
    train, validation, test = data["train"], data["validation"], data["test"]
    X, y, V = train["features"], train["labels"], validation["features"]
    scorers, params_used, seconds = {}, {}, {}

    # Heuristics read the candidate features; CF baselines read the plays. Every scorer gets
    # (features, candidates, feature engineer) so the same models can be re-scored later.
    scorers["random"] = lambda frame, candidates, fe: random_scores(candidates, seed)
    scorers["popularity"] = lambda frame, candidates, fe: popularity_scores(frame)
    scorers["trending_24h"] = lambda frame, candidates, fe: trending_scores(frame)
    scorers["category_history"] = lambda frame, candidates, fe: category_history_scores(frame)
    scorers["item_knn"] = lambda frame, candidates, fe: item_knn_scores(fe, candidates)
    started = time.perf_counter()
    svd = SVDRecommender(dimensions=engineer.dimensions, seed=seed).fit(
        splits["train"], engineer.user_ids, engineer.video_ids
    )
    seconds["svd_mf"] = time.perf_counter() - started
    scorers["svd_mf"] = lambda frame, candidates, fe: svd.score(candidates)
    params_used["svd_mf"] = {"dimensions": engineer.dimensions}

    # Learned linear baseline: small validation grid on the same rows and features.
    best = None
    for value in (0.1, 1.0):
        started = time.perf_counter()
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=value, max_iter=500, random_state=seed)
        ).fit(X, y)
        score = validation_ndcg(score_model(model, V), data)
        logger.info("logistic_regression C=%s validation NDCG@10 %.4f", value, score)
        if best is None or score > best[0]:
            best = (score, value, model, time.perf_counter() - started)
    _, value, linear, seconds["logistic_regression"] = best
    params_used["logistic_regression"] = {"C": value}
    scorers["logistic_regression"] = lambda frame, candidates, fe: score_model(
        linear, select(frame, columns)
    )

    # Final model: LightGBM LambdaRank tuned with Optuna on validation NDCG@10.
    trial_log = []

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        started = time.perf_counter()
        model = fit_lambdarank(params, data, columns, seed)
        score = validation_ndcg(model.predict(V), data)
        trial_log.append(
            {
                "trial": trial.number,
                "validation_ndcg@10": score,
                "best_iteration": int(model.best_iteration_),
                "seconds": time.perf_counter() - started,
                **params,
            }
        )
        logger.info("optuna trial %d: validation NDCG@10 %.4f", trial.number, score)
        return score

    started = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    best_params = {**study.best_params, "subsample_freq": 1}
    ranker = fit_lambdarank(best_params, data, columns, seed)
    seconds[FINAL_MODEL] = time.perf_counter() - started
    params_used[FINAL_MODEL] = {**best_params, "best_iteration": int(ranker.best_iteration_)}
    scorers[FINAL_MODEL] = lambda frame, candidates, fe: ranker.predict(select(frame, columns))

    # Ablations: same tuned parameters, one change each.
    ablation_models = {}
    for name, removed in ABLATIONS.items():
        kept = columns if removed is None else [c for c in columns if not removed(c)]
        objective_name = "binary" if removed is None else "lambdarank"
        model = fit_lambdarank(best_params, data, kept, seed, objective=objective_name)
        ablation_models[name] = (model, kept)
        scorers["ablation:" + name] = lambda frame, candidates, fe, model=model, kept=kept: (
            score_model(model, select(frame, kept))
        )

    # Validation metrics for every scorer (features already in memory).
    validation_scores = {
        name: scorer(validation["features"], validation["candidates"], engineer)
        for name, scorer in scorers.items()
    }

    def score_test(feature_engineer):
        """One pass over test in chunks, every scorer on the same rows; features not kept."""
        scores = {name: [] for name in scorers}
        queries = test["queries"]
        for first in range(0, queries.max() + 1, TEST_CHUNK_QUERIES):
            rows = (queries >= first) & (queries < first + TEST_CHUNK_QUERIES)
            candidates = test["candidates"][rows]
            frame = query_frame(feature_engineer, candidates, columns)
            for name, scorer in scorers.items():
                values = scorer(frame, candidates, feature_engineer)
                scores[name].append(np.asarray(values, dtype=np.float64))
        return {name: np.concatenate(parts) for name, parts in scores.items()}

    test_scores = score_test(engineer)
    frozen = {}
    if frozen_engineer is not None:
        for name, values in score_test(frozen_engineer).items():
            frozen[name] = per_query_metrics(values, test["labels"], test["queries"])[
                "ndcg@10"
            ].to_numpy()

    entries, per_query = {}, {}
    for name in scorers:
        val_q = per_query_metrics(
            validation_scores[name], validation["labels"], validation["queries"]
        )
        test_q = per_query_metrics(test_scores[name], test["labels"], test["queries"])
        per_query[name] = test_q["ndcg@10"].to_numpy()
        family, description = (
            FINAL_DESCRIPTION if name == FINAL_MODEL else BASELINES.get(name, ("Ablation", name))
        )
        entries[name] = {
            "family": family,
            "description": description,
            "validation": summarize(val_q),
            "test": summarize(test_q),
            "test_ci": {metric: bootstrap_ci(test_q[metric], seed=seed) for metric in test_q},
            "params": params_used.get(name, {}),
            "training_seconds": seconds.get(name),
        }
        logger.info(
            "%-40s val NDCG@10 %.4f | test NDCG@10 %.4f",
            name,
            entries[name]["validation"]["ndcg@10"],
            entries[name]["test"]["ndcg@10"],
        )

    # Best baseline = highest validation NDCG@10 among the baselines (chosen before testing).
    best_baseline = max(BASELINES, key=lambda name: entries[name]["validation"]["ndcg@10"])
    significance = paired_bootstrap_test(
        per_query[FINAL_MODEL], per_query[best_baseline], seed=seed
    )
    ablations = {
        name: {
            "removed_features": len(columns) - len(kept),
            "validation_ndcg@10": entries["ablation:" + name]["validation"]["ndcg@10"],
            "test_ndcg@10": entries["ablation:" + name]["test"]["ndcg@10"],
            "test_ndcg@10_ci": entries["ablation:" + name]["test_ci"]["ndcg@10"],
            "difference_vs_final": paired_bootstrap_test(
                per_query[FINAL_MODEL], per_query["ablation:" + name], seed=seed
            ),
        }
        for name, (_, kept) in ablation_models.items()
    }
    models = {name: entry for name, entry in entries.items() if not name.startswith("ablation:")}
    sensitivity = {}
    if frozen:
        frozen_best = max(BASELINES, key=lambda name: frozen[name].mean())
        sensitivity = {
            "description": "same trained models; test features read training plays only "
            "(histories frozen at the training cutoff)",
            "test_ndcg@10": {name: float(frozen[name].mean()) for name in models},
            "strongest_baseline_on_frozen_test": frozen_best,
            "final_vs_strongest_frozen_baseline": paired_bootstrap_test(
                frozen[FINAL_MODEL], frozen[frozen_best], seed=seed
            ),
        }
    importance = ranker.booster_.feature_importance(importance_type="gain")
    results = {
        "protocol": {
            "split": "chronological 64K / 16K / 20K plays (train / validation / test)",
            "query": "one first-time play (user had never played the video before)",
            "candidates_per_query": N_EVAL_NEGATIVES + 1,
            "negatives": "uniform from videos already uploaded at query time, excluding every "
            "video the user ever played",
            "train_negatives_per_query": train_negatives,
            "validation_queries_for_selection": len(np.unique(validation["queries"])),
            "features": "pre-watch only, point-in-time over the full play log (each request "
            "sees only earlier plays; learned inputs fit on training plays); observed watch "
            "features removed",
            "relevance": "played = 1, sampled = 0",
            "metric": "NDCG@10 (primary), HR@10, MRR, AUC; mean over queries",
            "confidence_intervals": "95% percentile bootstrap over test queries",
            "significance": "paired bootstrap over test queries, 10,000 resamples, two-sided",
            "model_selection": "validation NDCG@10 only; test scored once",
            "seed": seed,
        },
        "queries": {name: int(len(np.unique(part["queries"]))) for name, part in data.items()},
        "first_play_queries_available": {
            name: part["available_queries"] for name, part in data.items()
        },
        "candidate_rows": {name: int(len(part["labels"])) for name, part in data.items()},
        "ranking_features": len(columns),
        "models": models,
        "final_model": FINAL_MODEL,
        "best_baseline": best_baseline,
        "significance_vs_best_baseline": significance,
        "ablations": ablations,
        "frozen_history_sensitivity": sensitivity,
        "optuna": {
            "trials": trials,
            "best_params": best_params,
            "best_validation_ndcg@10": float(study.best_value),
            "trial_log": trial_log,
        },
        "feature_importance": sorted(
            (
                {"feature": column, "gain": float(value)}
                for column, value in zip(columns, importance)
            ),
            key=lambda row: -row["gain"],
        ),
        "feature_group_importance": grouped_importance(columns, importance),
        "seconds": time.perf_counter() - start_all,
    }
    return ranker, columns, results


def log_ranking_runs(results, tracking_uri=None):
    """One MLflow run per model, plus the Optuna trace on the final ranker's run."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location="experiments/artifacts/ranking")
    mlflow.set_experiment(EXPERIMENT)
    runs = []
    protocol = {f"protocol_{key}": str(value)[:250] for key, value in results["protocol"].items()}
    for name, entry in results["models"].items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"family": entry["family"], "final_model": name == FINAL_MODEL})
            mlflow.log_params({"model": name, **entry["params"], **protocol})
            for split in ["validation", "test"]:
                mlflow.log_metrics(
                    {
                        f"{split}_{metric.replace('@', '_at_')}": value
                        for metric, value in entry[split].items()
                    }
                )
            if name == FINAL_MODEL:
                significance = results["significance_vs_best_baseline"]
                mlflow.log_metrics(
                    {
                        "test_ndcg_at_10_gain_vs_best_baseline": significance["mean_difference"],
                        "paired_bootstrap_p_value": significance["p_value"],
                    }
                )
                for row in results["optuna"]["trial_log"]:
                    mlflow.log_metric(
                        "optuna_validation_ndcg_at_10", row["validation_ndcg@10"], step=row["trial"]
                    )
                mlflow.log_dict(
                    {
                        "feature_importance": results["feature_importance"][:50],
                        "feature_group_importance": results["feature_group_importance"],
                        "ablations": results["ablations"],
                        "significance_vs_best_baseline": significance,
                    },
                    "ranker_analysis.json",
                )
            runs.append({"run_id": run.info.run_id, "model": name, "test": entry["test"]})
    return runs
