import pandas as pd
from sklearn import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
import numpy as np
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics import classification


def get_data_split():

    data = pd.read_csv('/content/drive/MyDrive/dennis_ocr/archive/mnistA-Z0-9.csv').astype('float32')


    data.rename(columns={'0':'label'}, inplace=True)

    X = data.drop('label', axis = 1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    return X_train, X_test, y_train, y_test


def get_class_weights(y):

  weights = compute_class_weight(class_weight = 'balanced', classes = y.value_counts().index, y = y)

  return weights

def display(data, labels):

  #data, labels = utils.shuffle(data, labels)

  plt.figure(figsize=(9,9))
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)

    plt.imshow(data.iloc[i].values.reshape(28,28),interpolation='nearest', cmap='Greys')
    ax.title.set_text(labels[i])

  plt.show()

def show_img(img, label):
  plt.figure(figsize=(3,3))

  ax = plt.imshow(img.values.reshape(28,28),interpolation='nearest', cmap='Greys')
  print(label)

  plt.show()

def calculate_accuracy(output, target):
    predictions = torch.argmax(torch.softmax(output, dim = 1, dtype = None), dim = 1)
    predictions = predictions.cpu().numpy()
    target = target.cpu().numpy()

    return (np.sum(target == predictions)/len(predictions))

def save_model(model):
  torch.save(model.state_dict(), '/content/drive/MyDrive/dennis_ocr/ocr.pt')

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )






class Digits(Dataset):

  def __init__(self, X, y, transform = None):
    self.X = np.asarray(X)
    self.X = self.X.reshape(self.X.shape[0], 28, 28).astype('float32')
    #print(self.X.shape)
    #print(self.X[0].shape)
    self.X = torch.from_numpy(self.X).float()
    self.y = np.asarray(y)
    self.y = torch.from_numpy(self.y).long()
    self.transform = transform


  def __len__(self):
    return len(self.y)

  def __getitem__(self, index):
    target = self.X[index]
    #print(target.shape)
    target.unsqueeze_(0)
    target.repeat(3, 1, 1)
    label = self.y[index]
    if self.transform:
      target = self.transform(target)
    return target, label

def train(train_loader, model, criterion, optimizer, epoch, params, train_accs):
    torch.set_default_dtype(torch.float32)
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    temp_accs = []
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)

        target = target.to(params["device"], non_blocking=True).float()

        output = model(images)

        type(target)
        loss = criterion(output, target.long())
        accuracy = calculate_accuracy(output, target)
        temp_accs.append(accuracy)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )

    train_accs.append(np.mean(temp_accs))


def test_for_metric(model, params, test_dataloader, weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    preds = []
    actual = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(params["device"], non_blocking=True)
            output = model(images)

            preds.extend(torch.softmax(output, dim=1, dtype=None).cpu().numpy())

            for label in labels:
                actual.append(label)

    return np.asarray(preds), np.asarray(actual)

def get_data(X_train, X_test, y_train, y_test):


    train_transform = v2.Compose([
        v2.Resize((32, 32)),
        v2.RandomRotation(10),
        v2.RandomZoomOut(p=0.2),
        # v2.Random
        v2.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.2), ratio=(0.9, 1.1)),
        # v2.ToTensor(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = v2.Compose([
        v2.Resize((32, 32)),
        # v2.ToTensor(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    training = Digits(X_train, y_train, transform=train_transform)
    testing = Digits(X_test, y_test, transform=test_transform)

    train_dataloader = DataLoader(training, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(testing, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader

def evaluate(weights_path):
    preds, actual = test_for_metric(weights_path)

    roc = classification.MulticlassROC(36)
    fpr, tpr, thresholds = roc(torch.from_numpy(preds), torch.from_numpy(actual))

    fig, ax = roc.plot(score=True)
    ax.legend(loc='lower center', ncols=6)
    fig.set_size_inches(8, 8)

    fig.savefig(...)

    multi_accs = classification.MulticlassAccuracy(num_classes=36, average='none')
    multi_accs(torch.from_numpy(preds), torch.from_numpy(actual))

    fig, ax = multi_accs.plot()
    ax.legend(loc="lower center", ncols=3)
    fig.set_size_inches(8, 8)

    fig.savefig(...)

    confmat = classification.MulticlassConfusionMatrix(num_classes=36)
    confmat(torch.from_numpy(preds), torch.from_numpy(actual))

    fig, ax = confmat.plot()
    fig.set_size_inches(16, 16)

    fig.savefig(...)


X_train, X_test, y_train, y_test = get_data_split()

train_dataloader, test_dataloader = get_data(X_train, X_test, y_train, y_test)

params = {
    "model": "resnet50",
    #"device": "cuda", # for GPU
    "device": "cpu", # for CPU
    "lr": 0.0001,
    "batch_size": 128,
    "num_workers": 0,
    "epochs": 10,
}


model = getattr(models, params["model"])(pretrained=False, num_classes=36,)
model = model.to(params["device"])

criterion = nn.CrossEntropyLoss(weight = torch.from_numpy(get_class_weights(y_test)).float()).to(params["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)

torch.set_default_dtype(torch.float32)

train_accs = []

for i in range(params['epochs']):
  train(train_dataloader,  model, criterion, optimizer, i, params, train_accs)
  if train_accs[i] > 0.95 or i == (params['epochs'] - 1):
    save_model(model)
    break

evaluate('/content/drive/MyDrive/dennis_ocr/ocr.pt')


