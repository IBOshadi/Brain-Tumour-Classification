import json
import torch
from abc import abstractmethod
from datetime import datetime
from numpy import inf
from logger import WriterTensorboardX
import os
import time
import matplotlib.pyplot as plt
from logger.training_log import append_training_log_row
from utils import get_git_commit_hash, get_git_status, write_json
from pathlib import Path


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1 and self.device.type == 'cuda':
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        # setup visualization writer instance
        self.writer = WriterTensorboardX(config.log_dir, self.logger, cfg_trainer['tensorboardX'])

        # ========== Lists for plotting curves ==========
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        # ===============================================

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_curves(self):
        """Save training curves as images"""
        if len(self.train_losses) == 0:
            self.logger.warning("No metrics recorded to plot curves.")
            return

        epochs = range(1, len(self.train_losses) + 1)

        # ----- Loss Curve -----
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        if len(self.val_losses) > 0:
            plt.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'loss_curve.png'), dpi=150)
        plt.close()

        # ----- Accuracy Curve -----
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_accuracies, 'b-', label='Train Accuracy')
        if len(self.val_accuracies) > 0:
            plt.plot(epochs, self.val_accuracies, 'r-', label='Val Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'accuracy_curve.png'), dpi=150)
        plt.close()

        self.logger.info(f"Curves saved to {self.checkpoint_dir}")

    def _append_training_log(self, run_metadata, epoch_log):
        """Append one completed epoch to the repository-wide training CSV."""
        config = self.config.config
        optimizer_config = config.get('optimizer', {})
        data_args = config.get('data_processing', {}).get('args', {})
        row = {
            'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'run_type': 'training',
            'run_id': str(Path(self.checkpoint_dir)),
            'experiment_name': run_metadata['experiment_name'],
            'architecture': run_metadata['arch'],
            'freeze_strategy': run_metadata['freeze_strategy'],
            'commit_hash': run_metadata['commit_hash'],
            'git_dirty_at_start': run_metadata['git_dirty_at_start'],
            'checkpoint_path': '',
            'epoch': epoch_log['epoch'],
            'epochs_configured': run_metadata['epochs_configured'],
            'optimizer': optimizer_config.get('type'),
            'learning_rate': self.optimizer.param_groups[0].get('lr'),
            'batch_size': data_args.get('batch_size'),
            'training_seconds': epoch_log.get('training_seconds'),
            'validation_seconds': epoch_log.get('validation_seconds'),
            'epoch_training_seconds': epoch_log.get('epoch_training_seconds'),
            'epoch_validation_seconds': epoch_log.get('epoch_validation_seconds'),
            'epoch_elapsed_seconds': epoch_log.get('epoch_elapsed_seconds'),
            'epoch_seconds': epoch_log.get('epoch_seconds'),
            'run_elapsed_seconds': epoch_log.get('run_elapsed_seconds'),
            'loss': epoch_log.get('loss'),
            'accuracy': epoch_log.get('accuracy'),
            'val_loss': epoch_log.get('val_loss'),
            'val_accuracy': epoch_log.get('val_accuracy'),
            'test_seconds': '',
            'inference_seconds': '',
            'inference_ms_per_image': '',
            'test_samples': '',
            'test_loss': '',
            'test_accuracy': '',
            'metrics_json': json.dumps(epoch_log, sort_keys=True),
        }
        append_training_log_row(row)

    def train(self, trial=None):
        """
        Full training logic
        """
        not_improved_count = 0
        commit_hash = get_git_commit_hash()
        git_status = get_git_status()
        if commit_hash is None:
            self.logger.warning(
                "No Git commit is available; continuing training without commit provenance."
            )
        if git_status is None:
            self.logger.warning(
                "Git worktree status is unavailable; git_dirty_at_start will be null."
            )
        metrics_path = Path(self.checkpoint_dir) / 'metrics.json'
        run_metadata = {
            'commit_hash': commit_hash,
            'git_dirty_at_start': None if git_status is None else bool(git_status),
            'experiment_name': self.config.config.get('name'),
            'arch': self.config.config.get('arch', {}).get('type'),
            'freeze_strategy': self.config.config.get('freeze_strategy', 'none'),
            'epochs_configured': self.epochs,
            'status': 'running',
            'epoch_metrics': [],
        }
        write_json(run_metadata, metrics_path)
        run_started_at = time.perf_counter()
        cumulative_training_seconds = 0.0
        cumulative_validation_seconds = 0.0
        cumulative_epoch_seconds = 0.0

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_started_at = time.perf_counter()
            result = self._train_epoch(epoch)
            result['epoch_elapsed_seconds'] = time.perf_counter() - epoch_started_at

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            epoch_training_seconds = log.get('training_seconds', 0.0)
            epoch_validation_seconds = log.get('validation_seconds', 0.0)
            cumulative_training_seconds += epoch_training_seconds
            cumulative_validation_seconds += epoch_validation_seconds
            cumulative_epoch_seconds += log['epoch_elapsed_seconds']
            log['epoch_training_seconds'] = epoch_training_seconds
            log['epoch_validation_seconds'] = epoch_validation_seconds
            log['epoch_seconds'] = cumulative_epoch_seconds
            log['training_seconds'] = cumulative_training_seconds
            log['validation_seconds'] = cumulative_validation_seconds
            log['run_elapsed_seconds'] = time.perf_counter() - run_started_at

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            run_metadata['epoch_metrics'].append(log)
            write_json(run_metadata, metrics_path)
            self._append_training_log(run_metadata, log)

            if trial is not None:
                trial.report(log['val_accuracy'], step=epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

            # ========== Record metrics for curves ==========
            self.train_losses.append(log.get('loss', 0))
            self.train_accuracies.append(log.get('accuracy', 0))

            if 'val_loss' in log:
                self.val_losses.append(log['val_loss'])
            if 'val_accuracy' in log:
                self.val_accuracies.append(log['val_accuracy'])
            # ===============================================

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    if trial is not None:
                        trial.set_user_attr('best_epoch', epoch)
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn't improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            # Always save best model when it improves
            if best:
                self._save_checkpoint(epoch, save_best=True)

            # Also save periodic checkpoints
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)

        # Save curves only once at the end of training
        self._save_curves()
        run_metadata['status'] = 'completed'
        run_metadata['epochs_completed'] = len(run_metadata['epoch_metrics'])
        run_metadata['monitor_best'] = self.mnt_best
        run_metadata['run_summary'] = {
            'training_seconds': cumulative_training_seconds,
            'validation_seconds': cumulative_validation_seconds,
            'epoch_seconds': cumulative_epoch_seconds,
            'run_elapsed_seconds': time.perf_counter() - run_started_at,
        }
        write_json(run_metadata, metrics_path)
        append_training_log_row({
            'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'run_type': 'training_summary',
            'run_id': str(Path(self.checkpoint_dir)),
            'experiment_name': run_metadata['experiment_name'],
            'architecture': run_metadata['arch'],
            'freeze_strategy': run_metadata['freeze_strategy'],
            'commit_hash': run_metadata['commit_hash'],
            'git_dirty_at_start': run_metadata['git_dirty_at_start'],
            'epochs_configured': run_metadata['epochs_configured'],
            'training_seconds': cumulative_training_seconds,
            'validation_seconds': cumulative_validation_seconds,
            'epoch_seconds': cumulative_epoch_seconds,
            'run_elapsed_seconds': run_metadata['run_summary']['run_elapsed_seconds'],
            'metrics_json': json.dumps(run_metadata['run_summary'], sort_keys=True),
        })
        return self.mnt_best

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available (supports CUDA and MPS), move model into configured device
        """
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            list_ids = []
            self.logger.info("Using Mac GPU (MPS)")
        elif torch.cuda.is_available() and n_gpu_use > 0:
            n_gpu = torch.cuda.device_count()
            if n_gpu_use > n_gpu:
                self.logger.warning(f"Warning: Requested {n_gpu_use} GPUs but only {n_gpu} available.")
                n_gpu_use = n_gpu
            device = torch.device('cuda:0')
            list_ids = list(range(n_gpu_use))
            self.logger.info(f"Using CUDA GPU(s): {list_ids}")
        else:
            device = torch.device('cpu')
            list_ids = []
            self.logger.info("Using CPU")

        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, weights_only=False)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))