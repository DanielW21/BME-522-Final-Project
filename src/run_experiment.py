"""Run the configured CalMS21 development experiments."""

from datetime import datetime
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.ingest import ensure_split_manifest, load_development_data
from src.data.preprocessing import LABEL_NAMES
from src.evaluation.metrics import compare_model_runs, evaluate_predictions
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_metrics,
    plot_per_video_metrics,
)
from src.models.build import build_model


CONFIG_PATH = Path("src/configs/config.yaml")


def _save_results(results, run_dir):
    comparison = compare_model_runs(results)
    comparison.to_csv(run_dir / "comparison.csv", index=False)
    figure = plot_model_comparison(
        comparison,
        metric="macro_f1",
        output_path=run_dir / "figures" / "model_comparison_macro_f1.png",
    )
    plt.close(figure)

    for result in results:
        result_dir = run_dir / "runs" / f"{result['model']}_{result['feature_set']}"
        figure_dir = result_dir / "figures"
        figure_dir.mkdir(parents=True)

        metrics = dict(result["metrics"])
        metrics["validation_frames"] = int(result["per_video"]["frames"].sum())
        metrics["validation_videos"] = len(result["per_video"])
        (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        result["confusion_matrix"].to_csv(result_dir / "confusion_matrix.csv")
        result["per_video"].to_csv(result_dir / "per_video_metrics.csv", index=False)

        figures = [
            plot_confusion_matrix(
                result["confusion_matrix"],
                title=f"{result['model']} / {result['feature_set']}",
                output_path=figure_dir / "confusion_matrix.png",
            ),
            plot_per_class_metrics(
                metrics,
                metric="f1",
                output_path=figure_dir / "per_class_f1.png",
            ),
            plot_per_video_metrics(
                result["per_video"],
                metric="macro_f1",
                output_path=figure_dir / "per_video_macro_f1.png",
            ),
        ]
        for figure in figures:
            plt.close(figure)


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    project = config["project"]
    run = config["run"]

    ensure_split_manifest(
        project["training_json"],
        project["split_manifest"],
        project["random_seed"],
        project["validation_split"],
        project["split_candidates"],
    )

    run_dir = None
    if run["save_results"]:
        run_dir = Path(project["runs_dir"]) / datetime.now().strftime("%Y%m%dT%H%M%S%f")
        run_dir.mkdir(parents=True)
        shutil.copyfile(CONFIG_PATH, run_dir / "config.yaml")

    results = []
    for feature_set in run["feature_sets"]:
        feature_json = project["features_json"] if feature_set == "treba" else None
        data = load_development_data(
            project["training_json"],
            project["split_manifest"],
            feature_set,
            feature_json,
        )

        for model_name, enabled in run["models"].items():
            if not enabled:
                continue

            model = build_model(
                model_name,
                config["model_params"][model_name],
                config["preprocessing"],
                project["random_seed"],
            )
            model.fit(data["X_train"], data["y_train"])
            predictions = model.predict(data["X_validation"])
            validation_video_ids = np.asarray(data["validation_video_ids"], dtype=object)[
                data["validation_groups"]
            ]
            evaluation = evaluate_predictions(
                data["y_validation"],
                predictions,
                labels=list(LABEL_NAMES),
                label_names=LABEL_NAMES,
                video_ids=validation_video_ids,
            )

            result = {
                "model": model_name,
                "feature_set": feature_set,
                **evaluation,
            }
            results.append(result)

    if run_dir is not None:
        _save_results(results, run_dir)

    return results


if __name__ == "__main__":
    main()
