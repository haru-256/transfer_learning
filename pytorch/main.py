import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch.optim import lr_scheduler
import argparse
import pathlib
import numpy as np
from utils import train_model, weights_init, make_validation_dataset
from sklearn import model_selection
import pandas
import shutil
import datetime


if __name__ == '__main__':
    scine = datetime.datetime.now()
    # make parser
    parser = argparse.ArgumentParser(
        prog='classify mnist',
        usage='python train.py',
        description='description',
        epilog='end',
        add_help=True
    )
    # add argument
    parser.add_argument('-s', '--seed', help='seed',
                        type=int, required=True)
    parser.add_argument('-n', '--number', help='the number of experiments.',
                        type=int, required=True)
    parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 100',
                        type=int, default=100)
    parser.add_argument('-bs', '--batch_size', help='batch size. defalut value is 128',
                        type=int, default=128)
    parser.add_argument('-vs', '--val_size', help='validation dataset size. defalut value is 0.15',
                        type=float, default=0.15)
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0,'
                        ' -1 is means don\'t use gpu',
                        choices=[-1, 0, 1], type=int, default=0)
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)
    # 引数を解析する
    args = parser.parse_args()

    gpu = args.gpu
    batch_size = args.batch_size
    epoch = args.epoch
    seed = args.seed
    number = args.number  # number of experiments
    out = pathlib.Path("result_{0}/result_{0}_{1}".format(number, seed))

    # make directory
    pre = pathlib.Path(out.parts[0])
    for i, path in enumerate(out.parts):
        path = pathlib.Path(path)
        if i != 0:
            pre /= path
        if not pre.exists():
            pre.mkdir()
        pre = path

    # 引数の書き出し
    with open(out / "args.text", "w") as f:
        f.write(str(args))

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# epoch: {}'.format(epoch))
    print('# val size: {}'.format(args.val_size))
    print('# out: {}'.format(out), end="\n\n")

    if gpu == 0:
        device = torch.device("cuda:0")
    elif gpu == 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    # path to data directory
    data_dir = pathlib.Path('../../data/food-101/images').resolve()

    try:
        # data immigration for validation data
        val_dir, _ = make_validation_dataset(
            data_dir, seed=seed, test_size=args.val_size)

        # transform
        data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])
        }
        # load datasets
        image_datasets = {x: datasets.ImageFolder(data_dir / x,
                                                  transform=data_transform[x])
                          for x in ['train', 'val', 'test']}

        print("Train Dataset Size:", len(image_datasets['train']))
        print("Validation Dataset Size:", len(image_datasets['val']))
        print("Test Dataset Size:", len(image_datasets['test']), end="\n\n")

        # load resnet18 trained for ImageNet.
        model_ft = models.resnet18(pretrained=True)
        # replace output layer  with layer whose output features is 101.
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 101)
        # initilalize parameters of model
        model_ft.fc.apply(weights_init)
        model_ft = model_ft.to(device)
        # prepare optimizer & criterion
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                               step_size=7,
                                               gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        # train model
        best_model = train_model(model=model_ft, datasets=image_datasets, criterion=criterion,
                                 optimizer=optimizer_ft, device=device, num_epochs=epoch,
                                 batch_size=batch_size, out=out)
        # save best model
        torch.save(best_model.state_dict(), out / "best_model.pt")

    except Exception as error:
        import traceback
        traceback.print_exc()
    finally:
        # undo data immigration
        print("Undo data immigration")
        for path in val_dir.glob("*/*.jpg"):
            shutil.move(path, str(path).replace("val", "train"))
        print("Done!!", end="\n\n")

    print("Wall-Time:", datetime.datetime.now() - scine)
