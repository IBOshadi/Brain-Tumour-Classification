import argparse
import collections
import torch
import data_processing.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.freeze import apply_freeze_strategy
from parse_config import ConfigParser
from trainer import Trainer


def main(config, trial=None):
    logger = config.get_logger('train')

    # setup data_processing instances
    data_loader = config.initialize('data_processing', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    architecture = config['arch']['type']
    freeze_strategy = config.config.get('freeze_strategy', 'none')
    apply_freeze_strategy(model, architecture, freeze_strategy)
    logger.info("Freeze strategy: %s", freeze_strategy)
    logger.info(model)

    # ========== ADD CLASS WEIGHTS ==========
    # train_meta = config['data_processing']['args']['metas']
    # class_weights = get_class_weights(train_meta)
    
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # else:
    #     device = torch.device('cpu')
    # class_weights = class_weights.to(device)
    
    # # Create weighted loss
    # loss = module_loss.cross_entropy_loss(weight=class_weights)
    # =======================================

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler_config = config.config.get('lr_scheduler')
    if lr_scheduler_config:
        scheduler_args = dict(lr_scheduler_config.get('args', {}))
        step_size_up_epochs = scheduler_args.pop('step_size_up_epochs', None)
        if step_size_up_epochs is not None:
            scheduler_args['step_size_up'] = max(
                1, round(len(data_loader) * step_size_up_epochs)
            )
        scheduler_type = getattr(
            torch.optim.lr_scheduler, lr_scheduler_config['type']
        )
        lr_scheduler = scheduler_type(optimizer, **scheduler_args)
    else:
        lr_scheduler = None

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train(trial=trial)
    return trainer.mnt_best


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Brain MRI image classifier training')
    args.add_argument('-p', '--phase', default='train', type=str,
                        help='phase (default: train)')
    args.add_argument('-c', '--config', default='brain_tumor_train_config.json', type=str,
                        help='config file path (default: brain_tumor_train_config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_processing', 'args', 'batch_size')),
        CustomArgs(['--arch', '--architecture'], type=str, target=('arch', 'type')),
    ]
    config = ConfigParser(args, options)
    if config['arch']['type'] not in module_arch.__all__:
        raise ValueError(f"Unsupported architecture: {config['arch']['type']}")
    main(config)
