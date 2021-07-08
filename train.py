# -*- coding: utf-8 -*-
# @Brief: 训练脚本
from tensorflow.keras import optimizers, callbacks, utils, applications
from core.VOCdataset import VOCDataset
from nets.UNet import *
from core.losses import *
from core.metrics import *
from core.callback import *
import core.config as cfg
from evaluate import evaluate
import tensorflow as tf
import os
import cv2 as cv


def train_by_fit(model, epochs, train_gen, test_gen, train_steps, test_steps):
    """
    fit方式训练
    :param model: 训练模型
    :param epochs: 训练轮数
    :param train_gen: 训练集生成器
    :param test_gen: 测试集生成器
    :param train_steps: 训练次数
    :param test_steps: 测试次数
    :return: None
    """

    cbk = [
        callbacks.ModelCheckpoint(
            './weights/epoch={epoch:02d}_val_loss={val_loss:.04f}_miou={val_object_miou:.04f}.h5',
            save_weights_only=True),
    ]

    learning_rate = CosineAnnealingLRScheduler(epochs, train_steps, 1e-4, 1e-6, warmth_rate=0.1)
    optimizer = optimizers.Adam(learning_rate)
    lr_info = print_lr(optimizer)

    model.compile(optimizer=optimizer,
                  loss=crossentropy_with_logits,
                  metrics=[object_accuracy, object_miou, lr_info])

    model.fit(train_gen,
              steps_per_epoch=train_steps,
              validation_data=test_gen,
              validation_steps=test_steps,
              epochs=epochs,
              callbacks=cbk)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if not os.path.exists("weights"):
        os.mkdir("weights")

    model = UNet(cfg.input_shape, cfg.num_classes)
    model.summary()

    train_dataset = VOCDataset(cfg.train_txt_path, batch_size=cfg.batch_size, aug=True)
    test_dataset = VOCDataset(cfg.val_txt_path, batch_size=cfg.batch_size)

    train_steps = len(train_dataset) // cfg.batch_size
    test_steps = len(test_dataset) // cfg.batch_size

    train_gen = train_dataset.tf_dataset()
    test_gen = test_dataset.tf_dataset()

    train_by_fit(model, cfg.epochs, train_gen, test_gen, train_steps, test_steps)
