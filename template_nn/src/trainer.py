# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

trainer class for training a model

@author: tadahaya
"""
import torch

from .utils import save_checkpoint

class Trainer:
    """
    trainer class for training a model
    
    """
    def __init__(self, model, optimizer, criterion, exp_name, device, scheduler=None):
        """
        Args:
            model (nn.Module): model to be trained
            optimizer (torch.optim): optimizer
            criterion (torch.nn): loss function
            exp_name (str): experiment name
            device (str): device to be used for training
            scheduler (torch.optim.lr_scheduler, optional): learning rate scheduler
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.exp_name = exp_name
        self.device = device
        self.scheduler = scheduler


    def train(self, trainloader, testloader, num_epochs, save_model_every_n_epochs=0):
        """
        train the model for the specified number of epochs.

        Args:
            trainloader (torch.utils.data.DataLoader): training data loader
            testloader (torch.utils.data.DataLoader): test data loader
            num_epochs (int): number of epochs to train the model
            save_model_every_n_epochs (int): save the model every n epochs
        
        """
        # keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # training
        for i in range(num_epochs):
            train_loss = self.train_epoch(trainloader) # 1 epoch分のloss
            accuracy, test_loss = self.evaluate(testloader) # 1 epoch学習後のtestでのaccuracy, loss
            train_losses.append(train_loss) # train lossの記録
            test_losses.append(test_loss) # test lossの記録
            accuracies.append(accuracy) # accuracyの記録
            print(
                f"Epoch: {i + 1}, Train_loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
                )
            # schedulerがあればstep
            if self.scheduler is not None:
                self.scheduler.step()
            # モデルの保存処理
            if save_model_every_n_epochs > 0 and (i + 1) % save_model_every_n_epochs == 0 and i + 1 != num_epochs:
                print("> Save checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)
        return train_losses, test_losses, accuracies


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        total_loss = 0
        for data, label in trainloader:
            # batchをdeviceへ
            data, label = data.to(self.device), label.to(self.device)
            # 勾配を初期化
            self.optimizer.zero_grad()
            # forward
            output = self.model(data)
            # loss計算
            loss = self.criterion(output, label)
            # backpropagation
            loss.backward()
            # パラメータ更新
            self.optimizer.step()
            total_loss += loss.item() * len(data) # batch内のlossの合計, 後でdataset内の平均lossに変換する
        return total_loss / len(trainloader.dataset) # dataset内の平均lossを返す
    

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in testloader:
                # batchをdeviceへ
                data, label = data.to(self.device), label.to(self.device)
                # 予測
                output = self.model(data)
                # lossの計算
                loss = self.criterion(output, label)
                total_loss += loss.item() * len(data)
                # accuracyの計算
                predictions = torch.argmax(output, dim=1)
                correct += torch.sum(predictions == label).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss