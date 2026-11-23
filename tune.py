import argparse
import copy
import random
import re
from pathlib import Path

import numpy as np
import optuna
import torch

from parse_config import ConfigParser
from train import main as train
from utils import read_json, write_json


def _learning_rate_bounds(optimizer_type):
    name = optimizer_type.lower()
    if "adam" in name:
        return 1e-6, 3e-4
    if name == "sgd":
        return 1e-4, 1e-2
    return 1e-6, 1e-2


def _suggest_parameters(trial, config):
    optimizer = config["optimizer"]
    optimizer_args = optimizer["args"]
    low_lr, high_lr = _learning_rate_bounds(optimizer["type"])
    learning_rate = trial.suggest_float("learning_rate", low_lr, high_lr, log=True)

    optimizer_args["lr"] = learning_rate
    optimizer_args["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 1e-3, log=True
    )
    if "momentum" in optimizer_args:
        optimizer_args["momentum"] = trial.suggest_float("momentum", 0.8, 0.99)

    current_batch_size = config["data_processing"]["args"]["batch_size"]
    batch_sizes = sorted({
        max(1, current_batch_size // 4),
        max(1, current_batch_size // 2),
        current_batch_size,
    })
    config["data_processing"]["args"]["batch_size"] = trial.suggest_categorical(
        "batch_size", batch_sizes
    )

    scheduler = config.get("lr_scheduler")
    if scheduler and scheduler["type"] == "CyclicLR":
        scheduler["args"]["base_lr"] = learning_rate
        scheduler["args"]["max_lr"] = learning_rate * trial.suggest_float(
            "max_lr_factor", 2.0, 10.0
        )


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_study_name(name):
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    if not safe_name:
        raise ValueError("Study name must contain letters or numbers.")
    return safe_name


def main():
    parser = argparse.ArgumentParser(
        description="Tune brain MRI training hyperparameters with Optuna."
    )
    parser.add_argument("-c", "--config", default="brain_tumor_train_config.json")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum epochs per screening trial; retrain the winner for full epochs.",
    )
    parser.add_argument("--study-name", default="brain-mri-hpo")
    parser.add_argument("--storage", default="saved/optuna_study.db")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    study_name = _safe_study_name(args.study_name)
    base_config = read_json(Path(args.config))
    if base_config["trainer"].get("monitor") != "max val_accuracy":
        parser.error('The config must monitor "max val_accuracy" for HPO.')
    data_config = base_config["data_processing"]
    has_external_validation = (
        data_config.get("type") == "BrainMRIDataLoader"
        and (Path(data_config["args"]["data_dir"]) / "validation").is_dir()
    )
    if data_config["args"].get("validation_split", 0) <= 0 and not has_external_validation:
        parser.error("The config must reserve a non-empty validation split for HPO.")
    if "accuracy" not in base_config.get("metrics", []):
        parser.error('The config metrics must include "accuracy" for HPO.')

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
        _suggest_parameters(trial, trial_config)
        trial_config["trainer"]["epochs"] = args.epochs
        trial_config["trainer"]["save_period"] = args.epochs + 1

        experiment_name = "{}-trial-{}".format(study_name, trial.number)
        config = ConfigParser.from_config(trial_config, experiment_name)
        return train(config, trial=trial)

    study.optimize(objective, n_trials=args.trials)

    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("Optuna finished without a completed trial; no best config is available.")

    best_config = copy.deepcopy(base_config)
    best_params = study.best_params
    best_config["optimizer"]["args"]["lr"] = best_params["learning_rate"]
    best_config["optimizer"]["args"]["weight_decay"] = best_params["weight_decay"]
    if "momentum" in best_params:
        best_config["optimizer"]["args"]["momentum"] = best_params["momentum"]
    best_config["data_processing"]["args"]["batch_size"] = best_params["batch_size"]
    scheduler = best_config.get("lr_scheduler")
    if scheduler and scheduler["type"] == "CyclicLR":
        best_config["lr_scheduler"]["args"]["base_lr"] = best_params["learning_rate"]
        best_config["lr_scheduler"]["args"]["max_lr"] = (
            best_params["learning_rate"] * best_params["max_lr_factor"]
        )

    output_dir = Path(base_config["trainer"]["save_dir"]) / "optuna" / study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_config_path = output_dir / "best_config.json"
    write_json(best_config, best_config_path)

    print("Best validation accuracy: {:.6f}".format(study.best_value))
    print("Best parameters: {}".format(best_params))
    print("Best full-training config: {}".format(best_config_path))
    print("Train the selected config with: python train.py -c {}".format(best_config_path))


if __name__ == "__main__":
    main()
