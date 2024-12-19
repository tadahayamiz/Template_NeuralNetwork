# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

utils

@author: tadahaya
"""
import json, os, time
import matplotlib.pyplot as plt
import logging, datetime
import random
import numpy as np
import torch

from .models import MyNet # SHOULD BE CHANGED TO YOUR MODEL!!! Necessary for loading experiments

# Fix random seed for reproducibility
def fix_seed(seed:int=222, fix_gpu:bool=False):
    """
    Fix random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value for random number generators.
        fix_gpu (bool): If True, GPU-related randomness is also fixed. 
                        This may reduce performance but ensures reproducibility.
                        Default is False.

    Returns:
        None
    """
    # Fix seed for Python's built-in random module
    random.seed(seed)
    # Fix seed for NumPy
    np.random.seed(seed)
    # Fix seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Fix seed for PyTorch on GPU if GPU is available and requested
    if fix_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setup
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def to_logger(
        logger: logging.Logger, name:str='', obj=None,
        skip_keys=None, skip_hidden:bool=True
        ):
    """
    Log attributes of an object using the provided logger.

    Args:
        logger (logging.Logger): Logger instance.
        name (str): Name or header to log before object attributes.
        obj: Object whose attributes will be logged.
        skip_keys (set): Set of keys to skip when logging attributes.
        skip_hidden (bool): If True, skip attributes starting with '_'.

    """
    if skip_keys is None:
        skip_keys = set()
    logger.info(name)
    if obj is not None:
        for k, v in vars(obj).items():
            if k not in skip_keys:
                if skip_hidden and k.startswith('_'):
                    continue
                logger.info('  {0}: {1}'.format(k, v))


# Save and load experiments
def save_experiment(
        experiment_name, config, model, train_losses, test_losses,
        accuracies, classes, base_dir=""
        ):
    """
    save the experiment: config, model, metrics, and progress plot
    
    Args:
        experiment_name (str): name of the experiment
        config (dict): configuration dictionary
        model (nn.Module): model to be saved
        train_losses (list): training losses
        test_losses (list): test losses
        accuracies (list): accuracies
        classes (dict): dictionary of class names
        base_dir (str): base directory to save the experiment
    
    """
    if len(base_dir) == 0:
        base_dir = os.path.dirname(config["config_path"])
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    # save config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    # save metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
            'classes': classes,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    # plot progress
    plot_progress(
        experiment_name, train_losses, test_losses, config["num_epochs"], base_dir=base_dir
        )
    # save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    """
    save the model checkpoint

    Args:
        experiment_name (str): name of the experiment
        model (nn.Module): model to be saved
        epoch (int): epoch number
        base_dir (str): base directory to save the experiment

    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f"model_{epoch}.pt")
    torch.save(model.state_dict(), cpfile)


def get_component_list(model, optimizer, criterion, device, scheduler=None):
    """
    get the components of the model
    
    """
    components = {
    "model": model.__class__.__name__,
    "criterion": criterion.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "device": device.__class__.__name__,
    "scheduler": scheduler.__class__.__name__,
    }
    return components


def load_experiments(
        experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"
        ):
    outdir = os.path.join(base_dir, experiment_name)
    # load config
    configfile = os.path.join(outdir, "config.json")
    with open(configfile, 'r') as f:
        config = json.load(f)
    # load metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data["train_losses"]
    test_losses = data["test_losses"]
    accuracies = data["accuracies"]
    classes = data["classes"]
    # load model
    model = MyNet(config) # SHOULD BE CHANGED TO YOUR MODEL!!!
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile)) # checkpointを読み込んでから
    return config, model, train_losses, test_losses, accuracies, classes


def plot_progress(
        experiment_name:str, train_loss:list, test_loss:list, num_epoch:int,
        base_dir:str="experiments", xlabel="epoch", ylabel="loss"
        ):
    """ plot learning progress """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    epochs = list(range(1, num_epoch + 1, 1))
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 14
    ax.plot(epochs, train_loss, c='navy', label='train')
    ax.plot(epochs, test_loss, c='darkgoldenrod', label='test')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir + f'/progress_{ylabel}.tif', dpi=300, bbox_inches='tight')


def visualize_images(mydataset, indices:list=[], output:str="", nrow:int=3, ncol:int=4):
    """
    visualize the images in the given dataset
    
    """
    # indicesの準備
    assert len(indices) <= len(mydataset), "!! indices should be less than the total number of images !!"
    num_vis = np.min((len(mydataset), nrow * ncol))
    if len(indices) == 0:
        indices = torch.randperm(len(mydataset))[:num_vis]
    else:
        num_vis = len(indices)
    classes = mydataset.classes
    images = [np.asarray(mydataset[i][0]) for i in indices]
    labels = [mydataset[i][1] for i in indices]
    # 描画
    fig = plt.figure()
    for i in range(num_vis):
        ax = fig.add_subplot(nrow, ncol, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])
    plt.tight_layout()
    if len(output) > 0:
        plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.show()


# Timer related functions
def timer(start_time):
    """ Measure the elapsed time """
    elapsed_time = time.time() - start_time
    elapsed_hours = int(elapsed_time // 3600)  # hour
    elapsed_minutes = int((elapsed_time % 3600) // 60)  # min
    elapsed_seconds = int(elapsed_time % 60)  # sec
    res = f"{elapsed_hours:02}:{elapsed_minutes:02}:{elapsed_seconds:02}"
    print(f"Elapsed Time: {res}")
    return res


# Logger related functions
def init_logger(
        module_name:str, outdir:str='', tag:str='',
        level_console:str='info',level_file:str='info'
        ):
    """
    Initialize a logger with console and file handlers.

    Args:
        module_name (str): Name of the logger (e.g., module or script name).
        outdir (str): Directory to save the log file. Default is current directory.
        tag (str): Tag to identify the log file. Default is current timestamp.
        level_console (str): Logging level for console output. Default is 'info'.
        level_file (str): Logging level for file output. Default is 'info'.

    Returns:
        logging.Logger: Configured logger instance.

    """
    # Define logging levels
    level_dic = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG,
        'notset': logging.NOTSET
    }
    # Set default tag to timestamp if not provided
    if not tag:
        tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # Ensure outdir exists if specified
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        logfile = os.path.join(outdir, f'log_{tag}.txt')
    else:
        logfile = f'log_{tag}.txt'
    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels (fine-tuned in handlers)
    # Avoid adding duplicate handlers
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(logfile)
        fh.setLevel(level_dic[level_file])
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            "%Y%m%d-%H%M%S"
        ))
        logger.addHandler(fh)
        # Console handler
        sh = logging.StreamHandler()
        sh.setLevel(level_dic[level_console])
        sh.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            "%Y%m%d-%H%M%S"
        ))
        logger.addHandler(sh)
    return logger