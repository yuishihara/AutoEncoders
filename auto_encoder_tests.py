import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import mnist
from chainer.dataset import concat_examples
from convolutional_auto_encoder import ConvolutionalAutoEncoder
import matplotlib.pyplot as plt


use_gpu = True
gpu_id = -1
if use_gpu:
    gpu_id = 0


def train_model(model):
    print('training model: ', model.name)
    train, test = mnist.get_mnist(withlabel=True, ndim=1)

    x, t = train[0]
    print('train[0] label: ', t)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    # plt.show() # uncomment to show image

    batch_size = 128

    train_iter = iterators.SerialIterator(
        train, batch_size, repeat=True, shuffle=True)
    test_iter = iterators.SerialIterator(
        test, batch_size, repeat=False, shuffle=False)

    if use_gpu:
        model.to_gpu(gpu_id)

    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    max_epoch = 10
    while train_iter.epoch < max_epoch:
        train_batch = train_iter.next()
        image_train, _ = concat_examples(train_batch, gpu_id)
        image_train = image_train.reshape(-1, 1, 28, 28)
        # print('training image shape: ', image_train.shape)

        prediction_train = model(image_train)

        loss = F.mean_squared_error(image_train, prediction_train)
        # print('loss shape: ', loss.shape)
        model.cleargrads()
        loss.backward()

        optimizer.update()

        if train_iter.is_new_epoch:
            print('epoch:{:02d} train_loss:{:.04f} '.format(
                train_iter.epoch, float(cuda.to_cpu(loss.data))), end='')
            test_model(model, test_iter)
    return model


def test_model(model, test_iter):
    test_losses = []
    while True:
        test_batch = test_iter.next()
        image_test, _ = concat_examples(test_batch, gpu_id)
        image_test = image_test.reshape(-1, 1, 28, 28)
        prediction_test = model(image_test)

        loss_test = F.mean_squared_error(image_test, prediction_test)
        test_losses.append(cuda.to_cpu(loss_test.data))

        if test_iter.is_new_epoch:
            test_iter.epoch = 0
            test_iter.current_position = 0
            test_iter.is_new_epoch = False
            test_iter._pushed_position = None
            break
    print('val_loss:{:.04f}'.format(np.mean(test_losses)))


def save_model(path, model):
    serializers.save_npz(path, model)


def try_model(path, model):
    serializers.load_npz(path, model)

    _, test = mnist.get_mnist(withlabel=True, ndim=1)

    x, t = test[10]

    # change the size of minibatch
    x = x.reshape(1, 1, 28, 28)
    y = model(x).data
    print('x shape: ', x.shape)
    print('y shape: ', y.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(y.reshape(28, 28), cmap='gray')
    plt.title('decoded')
    plt.show()


def main():
    path = 'cae.model'
    model = ConvolutionalAutoEncoder()
    train_model(model)
    save_model(path, model)
    model = ConvolutionalAutoEncoder()
    try_model(path, model)


if __name__ == '__main__':
    main()
