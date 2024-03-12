import os.path
import cv2
import os
import numpy as np

from copy import deepcopy
from key_generator import unpickle


def block_segmentation(main_mat, sb_size):
    main_mat = deepcopy(main_mat)
    c, h, w = main_mat.shape
    col_blocks = h // sb_size
    row_blocks = w // sb_size

    block_list = []
    for i in range(row_blocks):
        for j in range(col_blocks):
            sub_mat = main_mat[:, sb_size * i: sb_size + (sb_size * i),
                      sb_size * j: sb_size + (sb_size * j)]

            block_list.append(sub_mat)

    return block_list


def block_integration(block_list, mb_size):
    c = block_list[0].shape[0]
    sb_size = block_list[0].shape[-1]
    cover = np.zeros((c, mb_size, mb_size), dtype=np.uint8)

    row_blocks = mb_size // sb_size
    col_blocks = mb_size // sb_size

    idx = 0
    for i in range(row_blocks):
        for j in range(col_blocks):
            cover[:, sb_size * i: sb_size + (sb_size * i),
            sb_size * j: sb_size + (sb_size * j)] = block_list[idx]
            idx += 1

    return cover


def block_shuffling(block_list, key_seq):
    shuffling_list = []

    for i in range(len(block_list)):
        block_index = key_seq[i]
        shuffling_list.append(block_list[block_index])

    return shuffling_list


def block_rotation(block, key):
    block_copy = deepcopy(block)
    block = np.rot90(block_copy, key, axes=[1, 2])
    return block


def block_flipping(block, key):
    block_copy = deepcopy(block)
    block = np.flip(block_copy, key)
    return block


def np_inversion(block, key):
    if key:
        block = 255. - block

    return block


def channel_shuffling(block, key):
    block_copy = deepcopy(block)
    r, g, b = block[0], block[1], block[2]

    channels_order = ((r, g, b), (r, b, g), (g, r, b), (g, b, r), (b, r, g), (b, g, r))
    block_copy[0], block_copy[1], block_copy[2] = channels_order[key]
    return block_copy


class ImgGenerator:
    def __init__(self, img_size, block_size, sub_block_size, key_sets):
        self.img_size = img_size
        self.b_size = block_size
        self.sb_size = sub_block_size
        self.key_sets = key_sets

    def encryption(self, plain_img):
        plain_img = cv2.resize(plain_img, (self.img_size, self.img_size))
        rgb_img = cv2.cvtColor(plain_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

        b_list = block_segmentation(rgb_img, self.b_size)
        for i in range(len(b_list)):
            b = b_list[i]
            sb_list = block_segmentation(b, self.sb_size)
            sb_list = block_shuffling(sb_list, self.key_sets["sb_shuffling"][i])
            for j in range(len(sb_list)):
                sb = sb_list[j]
                sb = block_rotation(sb, self.key_sets["sb_rotation"][i][j])
                sb = block_flipping(sb, self.key_sets["sb_flipping"][i][j])
                sb = np_inversion(sb, self.key_sets["sb_NPtrans"][i][j])
                sb = channel_shuffling(sb, self.key_sets["sb_c_shuffling"][i][j])

                sb_list[j] = sb

            b_list[i] = block_integration(sb_list, self.b_size)
        # b_list = self.block_shuffling(b_list, self.key_sets["mb_shuffling"])

        encrypted_img = block_integration(b_list, self.img_size).transpose(1, 2, 0)
        bgr_img = cv2.cvtColor(encrypted_img, cv2.COLOR_RGB2BGR)
        return bgr_img


def main():
    img_name = "lena.png"
    img_size = 224
    b_size = 16
    sb_size = 8

    img_path = f"imgs/plain_samples/{img_name}"
    saving_root = f"imgs/etc_samples/{b_size}_{sb_size}"
    if os.path.exists(saving_root) is False:
        print("Creating saving Folder..")
        os.makedirs(saving_root)

    single_key = unpickle(f"key_set/single/{b_size}_{sb_size}_dict")
    multiple_key = unpickle(f"key_set/multiple/{b_size}_{sb_size}_dict")

    etc_generator = ImgGenerator(img_size, b_size, sb_size, single_key)
    plain_img = cv2.imread(img_path)

    # each block in img encrypted with the same key
    com_etc = etc_generator.encryption(plain_img)
    print("single key used ETC img has saved in : ",
          cv2.imwrite(os.path.join(saving_root, f"com_{b_size}_{sb_size}_{img_name}"), com_etc))

    # each block in img encrypted with the different key
    etc_generator.key_sets = multiple_key
    ind_etc = etc_generator.encryption(plain_img)
    print("multiple keys used ETC img has saved in :",
          cv2.imwrite(os.path.join(saving_root, f"ind_{b_size}_{sb_size}_{img_name}"), ind_etc))


if __name__ == '__main__':
    main()
