import datetime
import copy
import pathlib
# import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn import model_selection, preprocessing
import pandas as pd
import chainercv
from chainercv import transforms
from chainer import dataset, training
from chainer import link
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import reporter


class PytorchLike_LabeledImageDataset(dataset.DatasetMixin):
    """
    datasets class that assumes file structure like Pytorch.

    Parameters
    -------------------------
    dir_path: pathlib.Path
        path to directory to hold the data.

    transform_list: list
        list to holds transformars that collable function.

    param_list: list of dictionary
        lisy of dictionary to pass to transform.
    """

    def __init__(self, dir_path, transform_list, param_list):
        self.transform_list = transform_list
        self.param_list = param_list
        if not dir_path.is_absolute():
            dir_path.resolve()
        # label
        labels = [path.parts[-1] for path in dir_path.glob('*')]
        self.le = preprocessing.LabelEncoder().fit(labels)

        # store paths and labels into list
        self._pairs = pd.DataFrame(
            [(path, path.parts[-2]) for path in dir_path.glob('*/*.jpg')], columns=["path", "label"])
        self._pairs.label = self.le.transform(self._pairs.label)

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, label = self._pairs.iloc[i]
        # return images whise datafoirmat is CHW
        image = chainercv.utils.read_image(path)
        for param, transform in zip(self.param_list, self.transform_list):
            image = transform(image, **param)
        # Normalize
        for ch, (mean, std) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
            image[ch] = (image[ch] - mean) / std
        label = np.array(label, dtype=np.int32)

        return image, label


class Classifier(link.Chain):

    """A simple classifier model.
    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.
    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (callable):
            Loss function.
            You can specify one of loss functions from
            :doc:`built-in loss functions </reference/functions>`, or
            your own loss function (see the example below).
            It should not be an
            :doc:`loss functions with parameters </reference/links>`
            (i.e., :class:`~chainer.Link` instance).
            The function must accept two argument (an output from predictor
            and its ground truth labels), and return a loss.
            Returned value must be a Variable derived from the input Variable
            to perform backpropagation on the variable.
        accfun (callable):
            Function that computes accuracy.
            You can specify one of evaluation functions from
            :doc:`built-in evaluation functions </reference/functions>`, or
            your own evaluation function.
            The signature of the function is the same as ``lossfun``.
        label_key (int or str): Key to specify label variable from arguments.
            When it is ``int``, a variable in positional arguments is used.
            And when it is ``str``, a variable in keyword arguments is used.
    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (callable):
            Loss function.
            See the description in the arguments for details.
        accfun (callable):
            Function that computes accuracy.
            See the description in the arguments for details.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.
    .. note::
        This link uses :func:`chainer.softmax_cross_entropy` with
        default arguments as a loss function (specified by ``lossfun``),
        if users do not explicitly change it. In particular, the loss function
        does not support double backpropagation.
        If you need second or higher order differentiation, you need to turn
        it on with ``enable_double_backprop=True``:
          >>> import chainer.functions as F
          >>> import chainer.links as L
          >>>
          >>> def lossfun(x, t):
          ...     return F.softmax_cross_entropy(
          ...         x, t, enable_double_backprop=True)
          >>>
          >>> predictor = L.Linear(10)
          >>> model = L.Classifier(predictor, lossfun=lossfun)
    """

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))

        super(Classifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args, **kwargs):
        """Computes the loss value for an input and label pair.
        It also computes accuracy and stores it to the attribute.
        Args:
            args (list of ~chainer.Variable): Input minibatch.
            kwargs (dict of ~chainer.Variable): Input minibatch.
        When ``label_key`` is ``int``, the correpoding element in ``args``
        is treated as ground truth labels. And when it is ``str``, the
        element in ``kwargs`` is used.
        The all elements of ``args`` and ``kwargs`` except the ground trush
        labels are features.
        It feeds features to the predictor and compare the result
        with ground truth labels.
        Returns:
            ~chainer.Variable: Loss value.
        """

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*args, **kwargs)
        self.loss = self.lossfun(self.y["prob"], t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y["prob"], t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


"""
class NormUpdater(training.updaters.StandardUpdater):
    def __init__(self, iterator, optimizer, normalize_param, converter=dataset.convert.concat_examples,
                 device=None, loss_func=None, loss_scale=None):
        self.normalize_param = normalize_param
        super.__init__(iterator, optimizer, converter=dataset.convert.concat_examples,
                       device=None, loss_func=None, loss_scale=None)

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        in_arrays[0][0] =

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays
"""


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
    df = pd.DataFrame(paths, columns=["path", "class"])
    _, val_data_path = model_selection.train_test_split(df.loc[:, "path"], test_size=test_size,
                                                        stratify=df["class"], random_state=seed)
    for path in val_data_path:
        class_dir = val_dir / path.parts[-2]  # クラスラベルのディレクトリ
        if not class_dir.exists():
            class_dir.mkdir()

        # 画像ファイルの移動
        shutil.move(path, val_dir / "/".join(path.parts[-2:]))
    print("Done")

    return val_dir, val_data_path
