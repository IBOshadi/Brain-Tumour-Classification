# Brain MRI Tumor Classification

A PyTorch image-classification project for identifying three types of brain tumors from MRI scans: **meningioma**, **glioma**, and **pituitary tumor**. The project provides preprocessing, training, testing, hyperparameter tuning, architecture search, experiment logging, and report generation.

![Training](images/training.PNG)

## Dataset and classes

The project uses folder-based MRI images in `scan_images/` and prepared folder splits in `processed_scan_images_mri/`.

| Split | Images |
|---|---:|
| Training | 2,144 |
| Validation | 460 |
| Test | 460 |
| **Total** | **3,064** |

Folder names map to the following classes:

| Folder | Class |
|---:|---|
| `1` | Meningioma |
| `2` | Glioma |
| `3` | Pituitary tumor |

## Features

- MRI preprocessing and train/validation/test splitting
- ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, SqueezeNet, and EfficientNet-B0
- CPU, NVIDIA CUDA, and Apple MPS device selection
- Pretrained ImageNet weights
- Training augmentation, weighted sampling, and cyclic learning-rate scheduling
- Accuracy and macro-F1 metrics
- Optuna hyperparameter tuning and architecture search
- Checkpoints, run configurations, TensorBoard logs, timing data, and CSV experiment logs
- Test reports with confusion matrices, classification reports, predictions, and ROC-AUC
- Optional Git commit and worktree provenance in run metadata

## Project structure

```text
base/                   Base data-loader, model, and trainer classes
data_processing/        MRI preprocessing and data-loader code
model/                  Network architectures, losses, metrics, and freeze strategies
trainer/                Training loop and checkpoint handling
logger/                 Logging, TensorBoard, and experiment-log utilities
notebooks/              Exploratory data-analysis notebooks
processed_scan_images_mri/
                        Prepared MRI train, validation, and test splits
saved/
                        Training runs, test runs, checkpoints, and Optuna studies
reports/                Generated evaluation reports
train.py                Train a model
test.py                 Evaluate a checkpoint
tune.py                 Tune hyperparameters with Optuna
automl.py               Search architectures and training settings
```

## Environment setup

The recorded environment is Python 3.14.8 on macOS, with PyTorch 2.15.0 and Torchvision 0.29.2. A compatible PyTorch build for your operating system and accelerator should be used.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `requirements-lock.txt` for the exact package snapshot:

```bash
python -m pip install -r requirements-lock.txt
```

Check the active runtime before an experiment:

```bash
python - <<'PY'
import platform, sys
import torch

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available())
PY
```

## Prepare the MRI data

Run every preprocessing cell in `data_processing/data_analysis.ipynb`. It creates:

- `processed_scan_images_mri/train/`
- `processed_scan_images_mri/validation/`
- `processed_scan_images_mri/test/`
- `processed_scan_images_mri/manifest.csv`
- `processed_scan_images_mri/normalization.json`

Do not train until `normalization.json` is present. The data loader requires exactly three class folders and validates that training and validation use the same class names.

## Training

The provided brain-tumor configuration is `brain_tumor_train_config.json`.

Train a selected architecture:

```bash
python train.py -c brain_tumor_train_config.json --arch ResNet18
```

Supported architecture overrides:

```text
ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, SqueezeNet, EfficientNetB0
```

Useful overrides:

```bash
python train.py -c brain_tumor_train_config.json --arch EfficientNetB0 --lr 0.0003 --bs 16
```

- `--arch` or `--architecture`: selects the model
- `--lr` or `--learning_rate`: overrides the optimizer learning rate
- `--bs` or `--batch_size`: overrides the batch size
- `-r` or `--resume`: resumes from a compatible saved checkpoint
- `-d`: sets visible GPU device indices

Training writes each run to `saved/training/<experiment>/<timestamp>/`. A run contains the model checkpoint(s), `train_config.json`, `metrics.json`, and logs. Epoch metrics are also appended to `training_log.csv`.

To resume, the checkpoint architecture and configuration must match the original run:

```bash
python train.py -c brain_tumor_train_config.json --arch EfficientNetB0 -r path/to/model_best.pth
```

## Hyperparameter tuning

Tune learning rate, weight decay, batch size, and SGD momentum using Optuna:

```bash
python tune.py -c brain_tumor_train_config.json --trials 20 --epochs 5
```

The tuning study is stored in `saved/optuna_study.db`. Each completed trial is pruned or ranked by validation accuracy. The selected configuration is written below `saved/optuna/` and can be used for full training:

```bash
python train.py -c saved/optuna/<study-name>/best_config.json
```

Screening trials are not intended to replace final evaluation. Increase `--epochs` when needed for a more reliable ranking.

## Automatic architecture selection

Compare supported architectures and tune architecture, freeze strategy, optimizer, learning rate, weight decay, batch size, scheduler factor, and epochs:

```bash
python automl.py -c brain_tumor_train_config.json --trials 20 --epochs 10
```

Search a subset of architectures:

```bash
python automl.py -c brain_tumor_train_config.json \
  --models ResNet18 ResNet50 EfficientNetB0 SqueezeNet \
  --trials 20 --epochs 10
```

Freeze strategies include:

- `none`: train the complete model
- `classifier_only`: train only the classification head
- `stem`: freeze the stem
- `first_stage`: freeze the stem and first backbone stage
- `first_two_stages`: freeze the stem and first two backbone stages

The search prints the best full-training configuration and writes it below `saved/automl/`. Train the result with the printed command.

## Testing

Evaluate a checkpoint with the test configuration:

```bash
python test.py -c brain_tumor_test_config.json \
  --arch EfficientNetB0 \
  -m "saved/training/<experiment>/<timestamp>/model_best.pth"
```

The architecture passed at test time must match the checkpoint. Testing writes a run snapshot under `saved/testing/` and appends test timing and metric fields to `training_log.csv`.

## Evaluation reports

`report.ipynb` generates per-run reporting from test predictions. Reports can include:

- Classification report
- Confusion matrix
- CSV and JSON result summaries
- Test predictions
- Balanced and macro metrics
- One-vs-rest macro ROC-AUC
- Test-loop, inference, and run elapsed time

Generated reports are stored in `reports/<run-id>/`.


## Reproducibility

For reproducible experiments:

1. Commit source and configuration changes before training.
2. Record or review the commit ID in the run's `metrics.json`.
3. Preserve the exact data split and `normalization.json`.
4. Use `requirements-lock.txt` for a dependency snapshot.
5. Keep each checkpoint and run configuration together.

Git provenance is written when available. The project is not currently a Git repository in every copied or exported environment; training will continue and record provenance as unavailable in that case.

Do not commit `.venv/`, datasets, processed data, model checkpoints, or `saved/` runs. See `.gitignore` for the current exclusions.

## Timing and experiment logs

`training_log.csv` records each epoch and completed test run. Timing fields include:

- Training, validation, and epoch seconds
- Cumulative run elapsed seconds
- Test-loop and forward-pass inference seconds
- Average inference milliseconds per image

Training and validation timing includes data loading and metric computation. Accelerator inference work is synchronized before cumulative inference time is measured.
