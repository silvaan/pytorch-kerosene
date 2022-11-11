import os
import sys
import logging
from collections import defaultdict
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        epochs,
        data_loaders,
        model_name=None,
        run_name=None,
        device=None,
        save_checkpoints='last',
        checkpoints_dir='checkpoints',
        use_logging=True,
        logging_dir='.',
        use_tensorboard=True,
        tensorboard_dir='runs'
    ):
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

        self.model_name = model_name or model.__class__.__name__
        self.run_name = run_name or self.model_name+'_'+datetime.now().strftime('%b-%d_%H-%M')

        # Training
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.data_loaders = data_loaders
        self.device = device

        # Checkpoints
        self.save_checkpoints = save_checkpoints
        self.checkpoints_dir = checkpoints_dir

        # Logging
        self.set_logger(logging_dir, run_name)
        self.logger.disabled = not use_logging

        # Tensorbaord
        self.use_tensorboard = use_tensorboard
        self.tensorboard_dir = tensorboard_dir
        if self.use_tensorboard:
            tb_logdir = os.path.join(self.tensorboard_dir, self.run_name)
            self.tb_writer = SummaryWriter(tb_logdir)

    def set_logger(self, logging_dir, run_name):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(sh)

        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(logging_dir, run_name + '.log'))
            fh.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
            self.logger.addHandler(fh)

    # HOOKS

    def before_train(self, *args, **kwargs):
        pass

    def after_train(self, *args, **kwargs):
        pass

    def before_epoch(self, *args, **kwargs):
        pass

    def after_epoch(self, *args, **kwargs):
        pass

    # CHECKPOINTS

    def save_checkpoint(self, epoch):
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        if self.save_checkpoints == 'all':
            filename = f'{self.run_name}_{epoch}.pt'
        elif self.save_checkpoints == 'last':
            filename = f'{self.run_name}.pt'
        else:
            filename = f'{self.run_name}_{epoch}.pt'
        
        checkpoint_filepath = os.path.join(self.checkpoints_dir, filename)
        model_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(model_dict, checkpoint_filepath)

    def load_checkpoint(self, run_name):
        if os.path.isfile(run_name):
            checkpoint_filepath = run_name
        else:
            checkpoint_filepath = os.path.join(self.checkpoints_dir,  f'{run_name}.pt')

        checkpoint = torch.load(checkpoint_filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # TRAINING

    def to_device(self, x):
        if torch.is_tensor(x):
            return x.to(self.device)
        elif isinstance(x, list) or isinstance(x, tuple):
            x_type = type(x)
            res = [self.to_device(item) for item in x]
            return x_type(res)
        elif isinstance(x, dict):
            res = {k: self.to_device(v) for k, v in x.items()}
            return res
        return x

    def dict2log(self, metrics):
        values = []
        for phase in metrics.keys():
            for metric in metrics[phase].keys():
                values.append(f'{phase}_{metric}: {metrics[phase][metric]:.5f}')
        output = ', '.join(values)
        return output

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def train(self):
        self.before_train()
        self.logger.info(f'Starting run {self.run_name}: {self.model_name} with {self.epochs} epochs in {self.device}...')
        phases = self.data_loaders.keys()
        training_metrics = {phase: defaultdict(list) for phase in phases}

        for epoch in range(self.epochs):
            self.before_epoch(epoch=epoch)
            epoch_metrics = {phase: {} for phase in phases}

            for phase in phases:
                temp_metrics = defaultdict(list)

                # Training phase
                if phase == 'train':
                    self.model.train()
                    for batch in tqdm(self.data_loaders[phase]):
                        batch = self.to_device(batch)
                        self.optimizer.zero_grad()
                        metrics = self.training_step(batch)
                        loss = metrics['loss']
                        loss.backward()
                        self.optimizer.step()

                        for metric, value in metrics.items():
                            temp_metrics[metric].append(value.detach().item())
                # Validation phase
                else:
                    self.model.eval()
                    with torch.no_grad():
                        for batch in self.data_loaders[phase]:
                            batch = self.to_device(batch)
                            with torch.no_grad():
                                metrics = self.validation_step(batch)

                            for metric, value in metrics.items():
                                temp_metrics[metric].append(value.detach().item())
                            
                # Average metrics across batches
                for metric, values in temp_metrics.items():
                    epoch_metrics[phase][metric] = sum(values)/len(values)
                    training_metrics[phase][metric].append(epoch_metrics[phase][metric])

                # Log metrics in Tensorboard
                if self.use_tensorboard:
                    for metric in epoch_metrics[phase]:
                        self.tb_writer.add_scalar(f'{phase}/{metric}', epoch_metrics[phase][metric], epoch)
                    
                    for name, weight in self.model.named_parameters():
                        self.tb_writer.add_histogram(name, weight, epoch)

            if self.save_checkpoints is not None:
                self.save_checkpoint(epoch)
            
            self.logger.info(self.dict2log(epoch_metrics))
            
            self.after_epoch(epoch=epoch, metrics=epoch_metrics)

        if self.use_tensorboard:
            self.tb_writer.close()

        self.logger.info('Finished training.')

        training_metrics = {k: dict(v) for k, v in training_metrics.items()}
        self.after_train(metrics=training_metrics)

        return training_metrics