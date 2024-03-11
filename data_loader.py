import os
import torchvision

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


def dataset2folder(dataset, saving_root):
    data_nums = dataset.__len__()
    imgs_id = [i for i in range(data_nums)]
    imgs_id_strings = [str(i).zfill(len(str(data_nums))) for i in imgs_id]

    for i, (img, label) in enumerate(dataset):
        img = img.resize((224, 224), resample=3)
        img_path = f"{label}_{imgs_id_strings[i]}.bmp"
        img.save(os.path.join(saving_root, img_path))
        print(i, img_path)


def get_transformer():
    print("return transformer for Testing")
    cifar10_mean = (0.5000, 0.5000, 0.5000)
    cifar10_std = (0.5000, 0.5000, 0.5000)
    transformer = T.Compose([
        T.Resize(224, interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(cifar10_mean, cifar10_std)
    ])

    return transformer


class Cifar(Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.img_name = os.listdir(data_root)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        img_path = os.path.join(self.data_root, img_name)

        img_data = Image.open(img_path)
        img_tensor = self.transform(img_data)
        label = float(img_name.split("_")[1])
        return img_tensor, label

    def __len__(self):
        return len(self.img_name)


def main():
    dataset = torchvision.datasets.CIFAR10("img/cifar10/source", train=False, download=True)
    dataset2folder(dataset, "img/cifar10/visiable_data")


if __name__ == '__main__':
    main()
