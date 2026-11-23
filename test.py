import argparse
import collections
import json
import torch
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import data_processing.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from logger.training_log import append_training_log_row, synchronize_device
from parse_config import ConfigParser
from utils import write_json


def main(config, resume):
    run_started_at = time.perf_counter()
    logger = config.get_logger('test')

    # setup data_processing instances
    data_loader = config.initialize('data_processing', module_data)

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    if not resume:
        raise ValueError("Provide a trained checkpoint with the --model option.")
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    # New
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    inference_seconds = 0.0
    test_started_at = time.perf_counter()

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            synchronize_device(device)
            inference_started_at = time.perf_counter()
            output = model(data)
            synchronize_device(device)
            inference_seconds += time.perf_counter() - inference_started_at

            # computing loss, metrics on test set
            batch_size = data.shape[0]
            total_loss += loss_fn(output, target).item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    test_seconds = time.perf_counter() - test_started_at
    run_elapsed_seconds = time.perf_counter() - run_started_at
    log = {'test_loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    log.update({
        'test_seconds': test_seconds,
        'run_elapsed_seconds': run_elapsed_seconds,
        'inference_seconds': inference_seconds,
        'inference_ms_per_image': inference_seconds / n_samples * 1000,
        'test_samples': n_samples,
    })
    logger.info(log)

    experiment_name = config.config.get('name')
    architecture = config.config.get('arch', {}).get('type')
    metric_values = {
        key: value
        for key, value in log.items()
        if key not in {
            'test_seconds',
            'run_elapsed_seconds',
            'inference_seconds',
            'inference_ms_per_image',
            'test_samples',
        }
    }
    append_training_log_row({
        'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'run_type': 'testing',
        'run_id': str(config.log_dir),
        'experiment_name': experiment_name,
        'architecture': architecture,
        'commit_hash': '',
        'git_dirty_at_start': '',
        'checkpoint_path': str(Path(resume)),
        'epoch': '',
        'epochs_configured': '',
        'optimizer': '',
        'learning_rate': '',
        'batch_size': getattr(data_loader, 'batch_size', ''),
        'training_seconds': '',
        'validation_seconds': '',
        'epoch_seconds': '',
        'run_elapsed_seconds': run_elapsed_seconds,
        'loss': '',
        'accuracy': '',
        'val_loss': '',
        'val_accuracy': '',
        'test_seconds': test_seconds,
        'inference_seconds': inference_seconds,
        'inference_ms_per_image': inference_seconds / n_samples * 1000,
        'test_samples': n_samples,
        'test_loss': total_loss / n_samples,
        'test_accuracy': log.get('accuracy', ''),
        'metrics_json': json.dumps(metric_values, sort_keys=True),
    })
    test_config_path = Path(config.log_dir) / 'test_config.json'
    if not test_config_path.is_file():
        raise FileNotFoundError(f"Testing run config snapshot not found: {test_config_path}")
    test_config = json.loads(test_config_path.read_text(encoding='utf-8'))
    test_config['results'] = {**log, 'test_samples': n_samples}
    write_json(test_config, test_config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Brain MRI image classifier testing')

    parser.add_argument('-p', '--phase', default='test', type=str,
                        help='phase (default: None)')
    parser.add_argument('-c', '--config', default='brain_tumor_test_config.json', type=str,
                      help='config file path')
    parser.add_argument('-m', '--model', default=None, type=str,
                        help='path to model (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--arch', '--architecture'], type=str, target=('arch', 'type')),
    ]
    config = ConfigParser(parser, options)
    args = parser.parse_args()
    main(config, args.model)
