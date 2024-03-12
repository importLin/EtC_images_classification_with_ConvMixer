import timm
import torch
import pickle
import logging
import os

from timm.utils import accuracy, AverageMeter
from torch.utils.data import DataLoader
from torch import nn
from data_loader import get_transformer, Cifar

from model_encryption import dict_remapping
from DynamicConv import conv_initialize


def unpickle(file):
    with open(file, 'rb') as fo:
        img_dict = pickle.load(fo)

    return img_dict


def create_logger(log_root, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = '[%(asctime)s] (%(name)s): %(levelname)s %(message)s'

    file_handler = logging.FileHandler(os.path.join(log_root, f"{name}.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    return logger


def val_one_epoch(model, data_loader, logger, device):
    print("testing mode activated!")
    model.eval().to(device)
    acc_meter = AverageMeter()
    with torch.no_grad():
        for i, (img_tensor, labels) in enumerate(data_loader):
            print(f"batch:{i}")
            img_tensor = img_tensor.to(device)
            labels = labels.to(device)

            out = model(img_tensor)
            batch_acc = accuracy(out, labels, topk=(1,))
            logger.info(f"Batch acc-{acc_meter.avg:.3f}")
            acc_meter.update(batch_acc[0].item(), labels.size(0))

    logger.info(f"Testing acc-{acc_meter.avg:.3f}")
    return acc_meter.avg


def main():
    img_size = 224
    patch_size = 16
    sb_size = 8
    out_channels = 1024
    encrypted_model_used = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformer = get_transformer()

    # model initialize
    model = timm.create_model("convmixer_1024_20_ks9_p14", pretrained=True, num_classes=10)
    model.stem[0] = nn.Conv2d(3, out_channels, kernel_size=patch_size, stride=patch_size)

    # loading and remapping the name space of weights used for PE layer
    pretrained_state_dict = torch.load("weights/baseline/convmixer_1024_20_ks9_p16_epoch125_9688.pth")
    pretrained_state_dict = dict_remapping(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    if encrypted_model_used:
        weights_root = f"weights/encrypted_weights/{patch_size}_{sb_size}"
        imgs_root = f"imgs/etc_dataset/{patch_size}_{sb_size}"
        log_root = "testing_log"

        model.stem[0] = conv_initialize(out_channels, img_size, patch_size, weights_root)

        logger = create_logger(log_root, f"encrypted_{patch_size}_{sb_size}")
        testing_set = Cifar(imgs_root, transformer)

    else:
        logger = create_logger("testing_log", f"baseline_p{patch_size}")
        testing_set = Cifar("imgs/CIFAR10_samples", transformer)

    testing_loader = DataLoader(
        testing_set,
        batch_size=10,
        shuffle=False,
        drop_last=True
    )
    #
    for epoch in range(1):
        acc = val_one_epoch(model, testing_loader, logger, device)


if __name__ == '__main__':
    main()
