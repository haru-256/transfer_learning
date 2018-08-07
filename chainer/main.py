import chainer
from chainer import training
from chainer.training import extensions
import chainer.links as L
from chainer import initializers
from chainer import training
from chainer.training import updaters
from chainer import optimizers
from chainer.training import extensions
import argparse
from chainer import training
from chainer import optimizers
from chainercv import transforms
import pathlib
import shutil
from utils import make_validation_dataset, PytorchLike_LabeledImageDataset, Classifier
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
                        type=int, default=64)
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
    print('# out: {}'.format(out))

    data_dir = pathlib.Path('../../data/food-101/images').resolve()
    train_dir_path = data_dir / 'train'
    test_dir_path = data_dir / 'test'

    try:
        # data immigration for validation data
        val_dir_path, _ = make_validation_dataset(
            data_dir, seed=seed, test_size=args.val_size)
        # load datasets
        train_transform_list = [
            transforms.random_sized_crop,
            transforms.resize
        ]
        train_param_list = [
            {},
            {"size": (224, 224)}
        ]
        val_transform_list = [
            transforms.resize,
            transforms.center_crop
        ]
        val_param_list = [
            {"size": (256, 256)},
            {"size": (224, 224)}
        ]
        test_transform_list = [
            transforms.resize,
            transforms.center_crop
        ]
        test_param_list = [
            {"size": (256, 256)},
            {"size": (224, 224)}
        ]
        train_datasets = PytorchLike_LabeledImageDataset(
            dir_path=train_dir_path, transform_list=train_transform_list, param_list=train_param_list)
        val_datasets = PytorchLike_LabeledImageDataset(
            dir_path=val_dir_path, transform_list=val_transform_list, param_list=val_param_list)
        test_datasets = PytorchLike_LabeledImageDataset(
            dir_path=test_dir_path, transform_list=test_transform_list, param_list=test_param_list)

        print("Train Dataset Size:", len(train_datasets))
        print("Validation Dataset Size:", len(val_datasets))
        print("Test Dataset Size:", len(test_datasets), end="\n\n")

        # make iterator
        train_iter = chainer.iterators.SerialIterator(
            train_datasets, batch_size)
        val_iter = chainer.iterators.SerialIterator(
            val_datasets, batch_size, repeat=False, shuffle=False
        )
        test_iter = chainer.iterators.SerialIterator(
            test_datasets, batch_size, repeat=False, shuffle=False)

        # build model
        model = chainer.links.VGG16Layers()
        num_ftrs = model.fc8.out_size
        # model.fc8 = L.Linear(in_size=None, out_size=101,
        #                      initialW=initializers.Normal(scale=0.02),
        #                      initial_bias=initializers.Normal(scale=0.02))
        model = Classifier(model)
        # model = L.Classifier(model)

        if gpu >= 0:
            # Make a specified GPU current
            chainer.backends.cuda.get_device_from_id(gpu).use()
            model.to_gpu()  # Copy the model to the GPU

        # make optimizer
        optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
        optimizer.setup(model)

        # make updater & set up trainer
        updater = updaters.StandardUpdater(
            train_iter, optimizer, device=gpu)
        trainer = training.Trainer(
            updater, stop_trigger=(epoch, 'epoch'), out=out)

        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.snapshot(
            filename='snapshot_epoch-{.updater.epoch}'))
        trainer.extend(extensions.snapshot_object(
            model.predictor, filename='model_epoch-{.updater.epoch}'))
        trainer.extend(extensions.Evaluator(val_iter, model, device=gpu))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
        trainer.extend(extensions.ProgressBar(update_interval=20))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
        trainer.extend(extensions.dump_graph('main/loss'))

        # run trainning
        trainer.run()

    except:
        Exception
        import traceback
        traceback.print_exc()
    finally:
        # undo data immigration
        print("Undo data immigration")
        for path in val_dir_path.glob("*/*.jpg"):
            shutil.move(path, str(path).replace("val", "train"))
        print("Done!!", end="\n\n")

    print("Wall-Time:", datetime.datetime.now() - scine)
