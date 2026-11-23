import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import optuna

from logger.training_log import append_training_log_row
from parse_config import ConfigParser
from train import main as train
from tune import _safe_study_name, _set_seed
from utils import read_json, write_json


AVAILABLE_MODELS = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "SqueezeNet",
    "EfficientNetB0",
]

FREEZE_STRATEGIES = [
    "none",
    "classifier_only",
    "stem",
    "first_stage",
    "first_two_stages",
]


def _suggest_training_parameters(trial, config, models, min_epochs, max_epochs):
    architecture = trial.suggest_categorical("architecture", models)
    freeze_strategy = trial.suggest_categorical(
        "freeze_strategy", FREEZE_STRATEGIES
    )
    optimizer_type = trial.suggest_categorical(
        "optimizer", ["Adam", "AdamW", "SGD"]
    )
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 5e-3, log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-6, 1e-3, log=True
    )
    max_lr_factor = trial.suggest_float(
        "max_lr_factor", 1.5, 8.0, log=True
    )

    current_batch_size = config["data_processing"]["args"]["batch_size"]
    batch_sizes = sorted({
        max(1, current_batch_size // 4),
        max(1, current_batch_size // 2),
        current_batch_size,
    })
    batch_size = trial.suggest_categorical("batch_size", batch_sizes)
    epochs = trial.suggest_int("epochs", min_epochs, max_epochs)

    config["arch"]["type"] = architecture
    config["freeze_strategy"] = freeze_strategy
    config["optimizer"] = {
        "type": optimizer_type,
        "args": {
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
    }
    if optimizer_type == "SGD":
        config["optimizer"]["args"].update({
            "momentum": 0.9,
            "nesterov": True,
        })
    config["data_processing"]["args"]["batch_size"] = batch_size
    config["trainer"]["epochs"] = epochs
    config["trainer"]["save_period"] = epochs + 1
    config["lr_scheduler"] = {
        "type": "CyclicLR",
        "args": {
            "base_lr": learning_rate,
            "max_lr": learning_rate * max_lr_factor,
            "step_size_up_epochs": max(0.5, epochs / 2),
            "cycle_momentum": False,
        },
    }

    return {
        "architecture": architecture,
        "freeze_strategy": freeze_strategy,
        "optimizer": optimizer_type,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_lr_factor": max_lr_factor,
        "batch_size": batch_size,
        "epochs": epochs,
    }


def main():
    automl_started_at = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Compare brain MRI model architectures and tune training settings with Optuna."
    )
    parser.add_argument("-c", "--config", default="brain_tumor_train_config.json")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum epochs to search per trial (defaults to the config epoch count).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=3,
        help="Minimum epoch count Optuna may choose.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        default=AVAILABLE_MODELS,
        help="Model architectures to compare.",
    )
    parser.add_argument(
        "--study-name",
        default="brain-tumor-automl",
        help="Base study name; the selected model set is appended automatically.",
    )
    parser.add_argument("--storage", default="saved/optuna_study.db")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")

    models = [model for model in AVAILABLE_MODELS if model in args.models]
    if not models:
        parser.error("Select at least one model with --models.")
    model_set = "-".join(model.lower() for model in models)
    base_config = read_json(Path(args.config))
    max_epochs = args.epochs or base_config["trainer"]["epochs"]
    if args.min_epochs < 1 or args.min_epochs > max_epochs:
        parser.error("--min-epochs must be between 1 and the maximum --epochs.")
    study_name = _safe_study_name(
        "{}-{}-freeze-optimizer-epoch-v2".format(args.study_name, model_set)
    )
    if base_config["trainer"].get("monitor") != "max val_accuracy":
        parser.error('The config must monitor "max val_accuracy" for AutoML.')
    data_config = base_config["data_processing"]
    has_external_validation = (
        data_config.get("type") == "BrainMRIDataLoader"
        and (Path(data_config["args"]["data_dir"]) / "validation").is_dir()
    )
    if data_config["args"].get("validation_split", 0) <= 0 and not has_external_validation:
        parser.error("The config must reserve a non-empty validation split for AutoML.")
    if "accuracy" not in base_config.get("metrics", []):
        parser.error('The config metrics must include "accuracy" for AutoML.')

    storage_path = Path(args.storage).resolve()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///" + storage_path.as_posix(),
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
        ),
    )

    def objective(trial):
        _set_seed(args.seed)
        trial_config = copy.deepcopy(base_config)
        _suggest_training_parameters(
            trial,
            trial_config,
            models,
            args.min_epochs,
            max_epochs,
        )

        experiment_name = "{}-trial-{}".format(study_name, trial.number)
        config = ConfigParser.from_config(trial_config, experiment_name)
        return train(config, trial=trial)

    study.optimize(objective, n_trials=args.trials)

    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError(
            "Optuna finished without a completed trial; no best config is available."
        )

    best_config = copy.deepcopy(base_config)
    best_params = study.best_params
    best_config["arch"]["type"] = best_params["architecture"]
    best_config["freeze_strategy"] = best_params["freeze_strategy"]
    best_config["optimizer"] = {
        "type": best_params["optimizer"],
        "args": {
            "lr": best_params["learning_rate"],
            "weight_decay": best_params["weight_decay"],
        },
    }
    if best_params["optimizer"] == "SGD":
        best_config["optimizer"]["args"].update({
            "momentum": 0.9,
            "nesterov": True,
        })
    best_config["data_processing"]["args"]["batch_size"] = best_params["batch_size"]
    best_epoch = study.best_trial.user_attrs.get(
        "best_epoch", best_params["epochs"]
    )
    best_config["trainer"]["epochs"] = best_epoch
    best_config["lr_scheduler"] = {
        "type": "CyclicLR",
        "args": {
            "base_lr": best_params["learning_rate"],
            "max_lr": best_params["learning_rate"] * best_params["max_lr_factor"],
            "step_size_up_epochs": max(0.5, best_epoch / 2),
            "cycle_momentum": False,
        },
    }

    output_dir = Path(base_config["trainer"]["save_dir"]) / "automl" / study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_config_path = output_dir / "best_config.json"
    write_json(best_config, best_config_path)
    automl_seconds = time.perf_counter() - automl_started_at
    automl_summary = {
        "study_name": study_name,
        "automl_seconds": automl_seconds,
        "completed_trials": len(completed_trials),
        "best_validation_accuracy": study.best_value,
        "best_trial_number": study.best_trial.number,
    }
    study.set_user_attr("last_invocation_automl_seconds", automl_seconds)
    write_json(automl_summary, output_dir / "automl_summary.json")
    write_json({
        "study_name": study_name,
        "best_validation_accuracy": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_parameters": best_params,
        "automl_seconds": automl_seconds,
    }, output_dir / "best_trial.json")
    append_training_log_row({
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_type": "automl_summary",
        "run_id": study_name,
        "experiment_name": base_config.get("name"),
        "automl_seconds": automl_seconds,
        "metrics_json": json.dumps(automl_summary, sort_keys=True),
    })

    print("Best validation accuracy: {:.6f}".format(study.best_value))
    print("Best parameters: {}".format(best_params))
    print("Selected training epochs (best validation epoch): {}".format(best_epoch))
    print("Total AutoML run time: {:.2f} seconds ({:.2f} minutes)".format(
        automl_seconds, automl_seconds / 60
    ))
    print("AutoML timing summary: {}".format(output_dir / "automl_summary.json"))
    print("Best training config: {}".format(best_config_path))
    print("Train the selected config with: python train.py -c {}".format(best_config_path))


if __name__ == "__main__":
    main()
