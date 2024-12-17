# -*- coding: utf-8 -*-
"""

CLI template

@author: tadahaya
"""
# packages installed in the current environment
import os
import datetime
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# original packages in src
from .src import utils
from .src import data_handler as dh
from .src.trainer import Trainer
from .src.models import MyNet

# === 基本的にタスクごとに変更 ===
# argumentの設定, 概ね同じセッティングの中で振りうる条件を設定
parser = argparse.ArgumentParser(description='CLI template')
parser.add_argument(
    'workdir',
    type=str,
    help='working directory that contains the dataset'
    )
parser.add_argument('--note', type=str, help='short note for this running')
parser.add_argument('--train', type=bool, default=True) # 学習ありか否か
parser.add_argument('--num_epochs', type=int, default=5) # epoch
parser.add_argument('--batch_size', type=int, default=128) # batch size
parser.add_argument('--lr', type=float, default=0.001) # learning rate
parser.add_argument('--save_every_n_epochs', type=float, default=0.001) # save model every n epochs
parser.add_argument('--seed', type=str, default=222) # seed
parser.add_argument('--num_workers', type=str, default=2) # num_workers, 基本2の倍数が望ましい

args = parser.parse_args() # Namespace object

# argsをconfigに変換
cfg = vars(args)

# seedの固定
utils.fix_seed(seed=args.seed, fix_gpu=False) # for seed control

# setup
now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
cfg["outdir"] = args.workdir + '/results/' + now # for output
if not os.path.exists(cfg["outdir"]):
    os.makedirs(cfg["outdir"])
cfg["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device

# データの準備
def prepare_data():
    """
    データの読み込み・ローダーの準備を実施
    加工済みのものをdataにおいておくか, argumentで指定したパスから呼び出すなりしてデータを読み込む
    inference用を読み込む際のものも用意しておくと楽
    例ではCIFAR10を使用
    
    """
    from torchvision.datasets import CIFAR10
    full_train_set = CIFAR10(root=cfg["workdir"], train=True, download=True, transform=None)
    general_transform = dh.get_general_transforms()
    train_set, test_set = dh.split_dataset(
        full_train_set, split_ratio=0.8, transform=general_transform, shuffle=True
        )
    train_loader = dh.prep_dataloader(
        train_set, cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True
        )
    test_loader = dh.prep_dataloader(
        test_set, cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
        )
    return train_loader, test_loader


# model等の準備
def prepare_model():
    """
    model, loss, optimizer, schedulerの準備
    argumentでコントロールする場合には適宜if文使うなり

    """
    model = MyNet(output_dim=10)
    model.to(cfg["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['num_epochs'], eta_min=0
        )
    return model, criterion, optimizer, scheduler


# === あまりタスクに依存しない ===
# 学習
def fit(model, train_loader, test_loader, criterion, optimizer, scheduler):
    """
    学習を実施

    Args:
        model (nn.Module): モデル
        train_loader (DataLoader): 学習用データローダ
        test_loader (DataLoader): テスト用データローダ
        criterion (torch.nn): 損失関数
        optimizer (torch.optim): 最適化手法
        scheduler (torch.optim.lr_scheduler): スケジューラ

    Returns:
        nn.Module: 学習後のモデル
        list: 学習時のlossのリスト
        list: テスト時のlossのリスト

    """
    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion,
        exp_name='exp', device=cfg["device"], scheduler=scheduler
        )
    train_loss, test_loss, accuracies = trainer.train(
        train_loader, test_loader, num_epochs=cfg["num_epochs"],
        save_model_every_n_epochs=cfg["save_every_n_epochs"]
        )
    return model, train_loss, test_loss, accuracies


def main():
    if args.train:
        # training mode
        start = time.time() # for time stamp
        # 1. data prep
        train_loader, test_loader = prepare_data()
        cfg["num_training_data"] = len(train_loader)
        cfg["num_test_data"] = len(test_loader)        
        # 2. model prep
        model, criterion, optimizer, scheduler = prepare_model()
        # 3. training
        model, train_loss, test_loss, accuracies = fit(
            model, train_loader, test_loader, criterion, optimizer, scheduler
            )
        # 4. modify config
        components = utils.get_component_list(model, optimizer, criterion, cfg["device"], scheduler)
        cfg.update(components) # update config

        print(cfg["device"])

        elapsed_time = utils.timer(start) # for time stamp
        cfg["elapsed_time"] = elapsed_time
        # 5. save experiment & config
        utils.save_experiment(
            experiment_name=now, config=cfg, model=model, train_losses=train_loss,
            test_losses=test_loss, accuracies=accuracies, classes=None, base_dir=cfg["outdir"]
            )
    else:
        # inference mode
        # データ読み込みをtestのみに変更などが必要
        pass


if __name__ == '__main__':
    main()