import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
from torch.utils.data import random_split, DataLoader
from torchvision import models, datasets, transforms
import numpy as np
import os
import time
import shutil
import matplotlib.pyplot as plt
from PIL import Image

# device
device = torch.device('cpu')

# model path
best_model_path = 'best_model_1.pt'

# preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# load data
data_dir = 'datasetv1'
dataset = datasets.ImageFolder(data_dir)

# 80% train 20% val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

class_names = dataset.classes

# def train_epoch(model, dataloaders, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     running_corrects = 0.0

#     for inputs, labels in dataloaders:
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         loss = criterion(outputs, labels)

#         optimizer.step()

#         running_loss += loss.item() * inputs.size(0)
#         running_corrects += torch.sum(preds == labels.data)
#     return running_loss, running_corrects

# def val_epoch(model, dataloaders, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     running_corrects = 0.0

#     with torch.no_grad():
#         for inputs, labels in dataloaders:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)
#         return running_loss, running_corrects

# def train_model(model, criterion, optimizer, scheduler, num_epoch=4):
#     since = time.time()

#     torch.save(model.state_dict(), best_model_path)
#     best_acc = 0.0
#     history = []

#     for epoch in range(num_epoch):
#         print(f'Epoch {epoch}/{num_epoch - 1}')
#         print('-' * 30)

#         train_loss, train_corrects = train_epoch(
#             model, dataloaders['train'], criterion, optimizer, device)
#         scheduler.step()
#         val_loss, val_corrects = val_epoch(
#             model, dataloaders['val'], criterion, device)

#         train_loss /= dataset_sizes['train']
#         train_acc = train_corrects.double() / dataset_sizes['train']
#         val_loss /= dataset_sizes['val']
#         val_acc = val_corrects.double() / dataset_sizes['val']

#         history.append([train_acc, val_acc, train_loss, val_loss])
#         print(f'Epoch {epoch} / {num_epoch - 1}: '
#               f'Train Loss {train_loss:.4f}, Acc {train_acc:.4f}, '
#               f'Val Loss {val_loss:.4f}, Acc {val_acc:.4f}')

#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), best_model_path)

#     time_elapsed = time.time() - since
#     print(f'\nTraining complete in {
#           time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val acc {best_acc:.4f}')

#     model.load_state_dict(torch.load(best_model_path))
#     return model, history


def main():
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        torch.save(model.state_dict(), best_model_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {
            time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(
            best_model_path, weights_only=True))
        return model

    model_ft = models.mobilenet_v2(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion,
                           optimizer_ft, exp_lr_scheduler, num_epochs=2)

    model_conv = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.classifier[1].in_features
    model_conv.classifier[1] = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.SGD(
        model_conv.classifier[1].parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion,
                             optimizer_conv, exp_lr_scheduler, num_epochs=2)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
    print('success')
