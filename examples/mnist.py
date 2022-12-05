import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as T
from pytorch_kerosene import Trainer


transforms = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.flatten())
])

dataset = MNIST(root='.', download=True, transform=transforms)
sizes = [int(s*len(dataset)) for s in [0.7, 0.15, 0.15]]
train_set, val_set, test_set = random_split(dataset, sizes)

data_loaders = {
    'train': DataLoader(train_set, batch_size=128, shuffle=True),
    'val': DataLoader(val_set, batch_size=128, shuffle=True)
}
test_loader = DataLoader(test_set, batch_size=128)


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
        X, _ = next(iter(self.data_loaders['train']))
        self.tb_writer.add_graph(self.model, X.to(self.device))

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
        
    def test_step(self, batch):
        X, y = batch
        pred = self.model(X)
        loss = self.criterion(pred, y)
        acc = (pred.argmax(-1) == y).sum() / X.shape[0]
        metrics = {'loss': loss, 'acc': acc}
        return metrics


model = MNISTClassifier(784, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

trainer = MyTrainer(
    run_name='model_test',
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    mixed_precision=True,
    epochs=10,
    data_loaders=data_loaders,
    save_checkpoints='saved/last',
    checkpoints_dir='saved/checkpoints',
    tensorboard_dir='saved/runs'
)

train_metrics = trainer.train()
test_metrics = trainer.test(test_loader)