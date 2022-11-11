import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as T
from pytorch_kerosene import Trainer


transforms = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.flatten())
])

dataset = MNIST(root='.', download=True, transform=transforms)
validation_split = 0.2
val_len = int(len(dataset)*validation_split)
train_len = len(dataset) - val_len
train_set, val_set = random_split(dataset, [train_len, val_len])

data_loaders = {
    'train': DataLoader(train_set, batch_size=32, shuffle=True),
    'val': DataLoader(val_set, batch_size=128, shuffle=True)
}


class MNISTClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        hidden_dim = input_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class MyTrainer(Trainer):
    def before_train(self):
        print(f'Starting training with {self.epochs} epochs in device {self.device}...')

    def training_step(self, batch):
        X, y = batch
        pred = self.model(X)
        loss = self.criterion(pred, y)
        acc = (pred.argmax(-1) == y).sum() / X.shape[0]
        metrics = {'loss': loss, 'acc': acc}
        return metrics

    def validation_step(self, batch):
        X, y = batch
        pred = self.model(X)
        loss = self.criterion(pred, y)
        acc = (pred.argmax(-1) == y).sum() / X.shape[0]
        metrics = {'loss': loss, 'acc': acc}
        return metrics


model = MNISTClassifier(784, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

trainer = MyTrainer(
    run_name='model_test',
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    epochs=3,
    data_loaders=data_loaders,
    save_checkpoints='last',
    checkpoints_dir='checkpoints',
    tensorboard_dir='runs'
)

metrics = trainer.train()