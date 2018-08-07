from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy
import pathlib
import torchvision
# import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn import model_selection
import pandas
from collections import OrderedDict
import json


# custom weights initialization
def weights_init(m):
    """
    Initialize

    Parameters
    ----------------------
    m: torch.nn.Module
        Module that means layer.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # Conv系全てに対しての初期化
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # initialize for BN
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        # initialize Linear
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.normal_(0, 0.02)


def train_model(model, datasets, optimizer, criterion, num_epochs=30, batch_size=128,
                device=None, scheduler=None, out=None):
    """
    train gan(generator, discriminator) with standard gan algorithm

    Parameters
    -----------------
    models: torch.nn.Module
        pre-trained model

    datasets: torch.utils.data.Dataset
        dataset of image

    optimizer: torch.optim
        optimizer for model

    criterion: torch.nn.Module
        function that calculates loss

    num_epochs: int
        number of epochs

    batch_size: int
        number of batch size

    device: torch.device

    out: pathlib.Path
        represent output directory

    Return
    -----------------------------
    model: torch.nn.Module
        best model
    """
    epochs = tqdm(range(num_epochs), desc="Epoch", unit='epoch')
    phases = ['train', 'val']
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # construct dataloader
    dataloader = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size,
                                                     shuffle=(phase == 'train'), num_workers=2)
                  for phase in ['train', 'val']}
    dataset_sizes = {phase: len(datasets[phase]) for phase in ['train', 'val']}
    # initialize log
    log = OrderedDict()
    # train loop
    since = datetime.datetime.now()
    for epoch in epochs:
        for phase in phases:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            train_loss = 0.0
            train_acc = 0.0
            # Iterate over data.
            iteration = tqdm(dataloader[phase],
                             desc="{} iteration".format(phase.capitalize()),
                             unit='iter')
            for inputs, labels in iteration:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # returns loss is mean_wise
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                train_loss += loss.item() * inputs.size(0)
                train_acc += torch.sum(preds == labels.data)

            epoch_loss = train_loss / dataset_sizes[phase]
            epoch_acc = train_acc.double().item() / dataset_sizes[phase]
            tqdm.write('Epoch: {:3d} Phase: {:>5}    Loss: {:.4f} Acc: {:.4f}'.format(
                epoch+1, phase.capitalize(), epoch_loss, epoch_acc))

            if phase == 'train':
                # preserve train log
                log["epoch_{}".format(epoch+1)] = OrderedDict(train_loss=epoch_loss,
                                                              train_acc=epoch_acc)
            elif phase == 'val':
                # preserve val log
                log["epoch_{}".format(epoch+1)].update(OrderedDict(val_loss=epoch_loss,
                                                                   val_acc=epoch_acc))
                if epoch_acc > best_acc:
                    # deep copy the model
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # save model by epoch
        torch.save(model.state_dict(), out /
                   "model_{}epoch.pt".format(epoch+1))
        tqdm.write("-"*60)

    time_elapsed = datetime.datetime.now() - since
    tqdm.write('Training complete in {}'.format(time_elapsed))
    tqdm.write('Best val Acc: {:4f}'.format(best_acc), end="\n\n")

    # load best model weights
    model.load_state_dict(best_model_wts)

    # if test set exists, calculate loss and accuracy for best model
    if "test" in datasets:
        model.eval()
        testloader = torch.utils.data.DataLoader(datasets["test"], batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        iteration = tqdm(testloader,
                         desc="Test iteration",
                         unit='iter')
        test_loss = 0.0
        test_acc = 0
        with torch.no_grad():
            for inputs, labels in iteration:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # returns loss is mean_wise
                loss = criterion(outputs, labels)
                # statistics
                test_loss += loss.item() * inputs.size(0)
                test_acc += torch.sum(preds == labels.data)

        test_loss = test_loss / len(datasets["test"])
        test_acc = test_acc.double().item() / len(datasets["test"])
        tqdm.write('Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(
            "Test", test_loss, test_acc), end="\n\n")
        # preserve test log
        log['test'] = OrderedDict(test_loss=test_loss,
                                  test_acc=test_acc)

    # save log
    with open(out / "log.json", "w") as f:
        json.dump(log, f, indent=4, separators=(',', ': '))

    return model


def make_validation_dataset(data_dir, seed=None, test_size=0.25):
    """
    make validation dataset using immigrating data

    Parameters
    ---------------------
    data_dir: pathlib.Path
        path to data directory

    seed: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
        to include in the test split. If int, represents the absolute number of test samples. 
        If None, the value is set to the complement of the train size. By default, 
        the value is set to 0.25. The default will change in version 0.21. It will remain 0.25
        only if train_size is unspecified, otherwise it will complement the specified train_size.

    Returns
    ------------------------
    val_dir: pathlib.Path
        path to validation datasets directory

    val_data_path: pandas.Series
        pandas.Series object whose each elements is path to validation data 
    """
    # data immigration for validation data
    print("make validation dataset")
    val_dir = (data_dir / "val").resolve()
    if not val_dir.exists():
        val_dir.mkdir()
        print("make directory for validation dataset:", val_dir)
    train_data_dir = (data_dir / "train").resolve()
    paths = [[path, path.parts[-2]] for path in train_data_dir.glob("*/*.jpg")]
    df = pandas.DataFrame(paths, columns=["path", "class"])
    _, val_data_path = model_selection.train_test_split(df.loc[:, "path"], test_size=test_size,
                                                        stratify=df["class"], random_state=seed)
    for path in val_data_path:
        class_dir = val_dir / path.parts[-2]  # クラスラベルのディレクトリ
        if not class_dir.exists():
            class_dir.mkdir()

        # 画像ファイルの移動
        shutil.move(path, val_dir / "/".join(path.parts[-2:]))
    print("Done!!", end="\n\n")

    return val_dir, val_data_path


def log_report(log, epoch, isTrain, **kwards):
    """
    make Oderdict object that contains log (e.g. train_loss, train_acc, val_train, val_test)

    Parameters
    -----------------------
    log: Oderdict
        Oderdict object that contains log

    epoch: int
        epoch

    isTrain: boolean
        whether mode is train or test

    **kwards
        parameters are pass to OderDict's constractor

    Returns
    ----------------------
    log: Oderdict
        Oderdict object that contains log
    """

    pass
