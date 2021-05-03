from dataloader import read_bci_data
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt


class EEGNet(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1, 1), padding=(0,25), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthconv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separconv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(736, 2, bias=True)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.depthconv(x)
        x = self.separconv(x)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x

def train_EEG(train_loader, test_loader, activations, device):
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0005
    epochs = 350
    df = pd.DataFrame()
    df['epoch'] = range(1, epochs+1)
    best_test_acc = {"ReLU":0, "LeakyReLU":0, "ELU":0}
    for name, activation in activations.items():
        model = EEGNet(activation)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        acc_train = list()
        acc_test = list()
        print("Training with activation:", name)
        for epoch in range(1, epochs+1):
            model.train()
            correct = 0
            total_loss = 0
            for _, (inputs, label) in enumerate(train_loader):
                inputs = inputs.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)
                predict = model(inputs)
                loss = criterion(predict, label)
                total_loss += loss.item()
                
                pred = predict.argmax(dim=1)
                for i in range(len(label)):
                    if pred[i] == label[i]:
                        correct +=1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss = total_loss / float(len(train_loader.dataset))
            correct = 100. * correct / float(len(train_loader.dataset))
            acc_train.append(correct)
            if epoch%10==0:
                print(f'epoch{epoch:>3d}  loss:{total_loss:.4f}  acc:{correct:.3f}%')
            model.eval()
            correct_test = evaluate(model, test_loader, device)
            if correct_test > best_test_acc[name]:
                best_test_acc[name] = correct_test
            acc_test.append(correct_test)
            if epoch%10==0:
                print(f'epoch{epoch:>3d}  acc_test:{correct_test:.2f}%')

        df[name+"_train"] = acc_train
        df[name+"_test" ] = acc_test
    return df, best_test_acc

def train_Deep(train_loader, test_loader, activations, device):
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    epochs = 350
    df = pd.DataFrame()
    df['epoch'] = range(1, epochs+1)
    best_test_acc = {"ReLU":0, "LeakyReLU":0, "ELU":0}
    for name, activation in activations.items():
        model = DeepConvNet(activation)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        acc_train = list()
        acc_test = list()
        print("Training with activation:", name)
        for epoch in range(1, epochs+1):
            model.train()
            correct = 0
            total_loss = 0
            for _, (inputs, label) in enumerate(train_loader):
                inputs = inputs.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)
                predict = model(inputs)
                loss = criterion(predict, label)
                total_loss += loss.item()
                
                pred = predict.argmax(dim=1)
                for i in range(len(label)):
                    if pred[i] == label[i]:
                        correct +=1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss = total_loss / float(len(train_loader.dataset))
            correct = 100. * correct / float(len(train_loader.dataset))
            acc_train.append(correct)
            if epoch%10==0:
                print(f'epoch{epoch:>3d}  loss:{total_loss:.4f}  acc:{correct:.3f}%')
            model.eval()
            correct_test = evaluate(model, test_loader, device)
            if correct_test > best_test_acc[name]:
                best_test_acc[name] = correct_test
            acc_test.append(correct_test)
            if epoch%10==0:
                print(f'epoch{epoch:>3d}  acc_test:{correct_test:.2f}%')

        df[name+"_train"] = acc_train
        df[name+"_test" ] = acc_test
    return df, best_test_acc

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    for _, (inputs, label) in enumerate(test_loader):
        inputs = inputs.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        predict = model(inputs)
        pred = predict.argmax(dim=1)
        for i in range(len(label)):
            if pred[i] == label[i]:
                correct +=1

    correct = 100. * correct / float(len(test_loader.dataset))
    return correct

def plot_EEG(dataframe):
    plt.figure(figsize=(10,5))
    plt.title("Activation function comparision(EEGNet)", fontsize=18)
    plt.ylabel("Accuracy(%)", fontsize=18)
    plt.xlabel("Epoch", fontsize=18)
    for name in dataframe.columns[1:]:
        plt.plot('epoch', name, data=dataframe)
    plt.legend(loc="lower right")
    # plt.show()

def plot_Deep(dataframe):
    plt.figure(figsize=(10,5))
    plt.title("Activation function comparision(DeepConvNet)", fontsize=18)
    plt.ylabel("Accuracy(%)", fontsize=18)
    plt.xlabel("Epoch", fontsize=18)
    for name in dataframe.columns[1:]:
        plt.plot('epoch', name, data=dataframe)
    plt.legend(loc="lower right")
    # plt.show()

class DeepConvNet(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(DeepConvNet, self).__init__()
        out_channels = [25, 25, 50, 100, 200]
        kernel_size = [(1,5), (2,1), (1,5), (1,5), (1,5)]
        self.conv0 = nn.Conv2d(1, out_channels[0], kernel_size=kernel_size[0])
        for i in range(1, len(out_channels)):
            setattr(self,'conv'+str(i), nn.Sequential(
                nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=kernel_size[i]),
                nn.BatchNorm2d(out_channels[i]),
                activation,
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            ))
        self.classify = nn.Linear(8600, 2)
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x

if __name__== "__main__":
    print(torch.__version__)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_data, train_label, test_data, test_label = read_bci_data()
    dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
    train_dataset = DataLoader(dataset, batch_size=64, shuffle=True)
    # for _, (inputs, label) in enumerate(train_dataset):
    #     print("label:", label)
    dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
    test_dataset = DataLoader(dataset, batch_size=64, shuffle=True)
    
    activations = {"ReLU":nn.ReLU(), "LeakyReLU":nn.LeakyReLU(), "ELU":nn.ELU()}
    print("EEGNet")
    data, best_test_acc = train_EEG(train_dataset, test_dataset, activations, device)
    plot_EEG(data)
    for name, acc in best_test_acc.items():
        print(name, " best accuracy EEG: ", acc, "%")

    print("DeepConvNet")
    data_deep, best_test_acc_deep = train_Deep(train_dataset, test_dataset, activations, device)
    plot_Deep(data_deep)
    for name, acc in best_test_acc_deep.items():
        print(name, " best accuracy Deep: ", acc, "%")

    