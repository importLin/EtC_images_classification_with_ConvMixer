import os.path
import cv2
import os
from img_encryption_sample import ImgGenerator
from key_generator import unpickle


def main():
    dataset_root = "imgs/CIFAR10_samples"
    img_names = os.listdir(dataset_root)

    img_size = 224
    b_size = 16
    sb_size = 8

    saving_root = f"imgs/etc_dataset/{b_size}_{sb_size}"
    if os.path.exists(saving_root) is False:
        print("Creating saving Folder..")
        os.makedirs(saving_root)

    multiple_key = unpickle(f"key_set/multiple/{b_size}_{sb_size}_dict")
    etc_generator = ImgGenerator(img_size, b_size, sb_size, multiple_key)

    for i in range(len(img_names)):
        current_img = cv2.imread(os.path.join(dataset_root, img_names[i]))
        etc_img = etc_generator.encryption(current_img)
        print("ETC img saving state:",
              cv2.imwrite(os.path.join(saving_root, f"{img_names[i]}"), etc_img))


if __name__ == '__main__':
    main()
