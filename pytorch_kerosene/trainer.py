import os
from collections import defaultdict
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        run_name,
        model,
        criterion,
        optimizer,
        epochs,
        data_loaders,
        device=None,
        checkpoints_dir='checkpoints',
        save_checkpoints='last'
    ):
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

        self.run_name = run_name
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.data_loaders = data_loaders
        self.device = device
        self.checkpoints_dir = checkpoints_dir
        self.save_checkpoints = save_checkpoints

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
        else:
            filename = f'{self.run_name}.pt'
        
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

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def train(self):
        self.before_train()
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
                    for batch in self.data_loaders[phase]:
                        batch = self.to_device(batch)
                        with torch.no_grad():
                            metrics = self.validation_step(batch)

                        for metric, value in metrics.items():
                            temp_metrics[metric].append(value.detach().item())
                            
                # Average metrics across batches
                for metric, values in temp_metrics.items():
                    epoch_metrics[phase][metric] = sum(values)/len(values)
                    training_metrics[phase][metric].append(sum(values)/len(values))

            self.save_checkpoint(epoch)
            self.after_epoch(epoch=epoch, metrics=epoch_metrics)

        training_metrics = {k: dict(v) for k, v in training_metrics.items()}
        self.after_train(metrics=training_metrics)

        return training_metrics