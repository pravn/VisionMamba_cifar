import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM
from einops.layers.torch import Reduce
import os

import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt

import pytorch_lightning as pl 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


from model_cifar import Vim 

import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
#from dataset_imagenet import ImagenetDataset
import tensorboard 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ViMamba(pl.LightningModule):

    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = Vim(**model_kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
        
def train_model(train_loader, val_loader, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ViMamba"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=200,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = ""
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = ViMamba.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
        trainer.fit(model, train_loader, val_loader)
        model = ViMamba.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    else:
        pl.seed_everything(42) # To be reproducable
        model = ViMamba(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = ViMamba.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}

    return model, result


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

CHECKPOINT_PATH = "./saved_models_cifar"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

#train_dataset = ImagenetDataset(500, 'train')
#train_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

#val_dataset = ImagenetDataset(100, 'validation')
#val_loader = data.DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=4)

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                     ])
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                     ])
# Loading the training dataset. We need to split it into a training and validation part
# We need to do a little trick because the validation set should not use the augmentation.
train_dataset = CIFAR10(root='/home/ubuntu/VisionMamba/CIFAR10', train=True, transform=train_transform, download=True)
val_dataset = CIFAR10(root='/home/ubuntu/VisionMamba/CIFAR10', train=True, transform=test_transform, download=True)
#pl.seed_everything(42)
train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
#pl.seed_everything(42)
_, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
test_set = val_set

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
test_loader = val_loader 



model, results = train_model(train_loader, val_loader, model_kwargs={
    'dim': 64,
    'dt_rank': 16,
    'dim_inner': 64,
    'd_state': 64,
    'num_classes': 10,
    'image_size': 32,
    'patch_size': 4,
    'channels': 3,
    'dropout': 0.1,
    'depth': 4}, lr=3e-4)