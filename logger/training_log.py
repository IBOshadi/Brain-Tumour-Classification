import csv
import os
from pathlib import Path


TRAINING_LOG_PATH = Path(__file__).resolve().parents[1] / 'training_log.csv'
TRAINING_LOG_FIELDS = [
    'recorded_at',
    'run_type',
    'run_id',
    'experiment_name',
    'architecture',
    'freeze_strategy',
    'commit_hash',
    'git_dirty_at_start',
    'checkpoint_path',
    'epoch',
    'epochs_configured',
    'optimizer',
    'learning_rate',
    'batch_size',
    'training_seconds',
    'validation_seconds',
    'epoch_training_seconds',
    'epoch_validation_seconds',
    'epoch_elapsed_seconds',
    'epoch_seconds',
    'run_elapsed_seconds',
    'loss',
    'accuracy',
    'val_loss',
    'val_accuracy',
    'test_seconds',
    'inference_seconds',
    'inference_ms_per_image',
    'automl_seconds',
    'test_samples',
    'test_loss',
    'test_accuracy',
    'metrics_json',
]


def append_training_log_row(row):
    """Append a training or testing record, upgrading older CSV headers safely."""
    TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TRAINING_LOG_FIELDS)

    if TRAINING_LOG_PATH.is_file() and TRAINING_LOG_PATH.stat().st_size:
        with TRAINING_LOG_PATH.open('r', newline='', encoding='utf-8') as log_file:
            reader = csv.DictReader(log_file)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)

        fieldnames.extend(field for field in existing_fields if field not in fieldnames)
        if existing_fields != fieldnames:
            temporary_path = TRAINING_LOG_PATH.with_suffix('.csv.tmp')
            with temporary_path.open('w', newline='', encoding='utf-8') as log_file:
                writer = csv.DictWriter(log_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
            os.replace(temporary_path, TRAINING_LOG_PATH)

    with TRAINING_LOG_PATH.open('a', newline='', encoding='utf-8') as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if log_file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        log_file.flush()


def synchronize_device(device):
    """Wait for queued accelerator work to finish when measuring elapsed time."""
    if device.type == 'cuda':
        import torch

        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        import torch

        torch.mps.synchronize()
