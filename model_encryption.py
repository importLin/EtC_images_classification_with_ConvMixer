import torch
import timm
import os

from key_generator import unpickle
# from block_operation import ModelBlockOp
from copy import deepcopy
from torch import nn


def mat_segmentation(main_mat, sb_size):
    main_mat = deepcopy(main_mat)
    k, c, h, w = main_mat.shape
    col_blocks = h // sb_size
    row_blocks = w // sb_size

    block_list = []
    for i in range(row_blocks):
        for j in range(col_blocks):
            sub_mat = main_mat[:, :, sb_size * i: sb_size + (sb_size * i),
                      sb_size * j: sb_size + (sb_size * j)]

            block_list.append(sub_mat)

    return block_list


def mat_integration(block_list, obj_shape):
    sb_size = block_list[0].shape[-1]

    k, c, h, w = obj_shape
    cover = torch.zeros((k, c, h, w))

    row_blocks = h // sb_size
    col_blocks = w // sb_size

    idx = 0
    for i in range(row_blocks):
        for j in range(col_blocks):
            cover[:, :, sb_size * i: sb_size + (sb_size * i),
            sb_size * j: sb_size + (sb_size * j)] = block_list[idx]
            idx += 1

    return cover


def mat_shuffling(block_list, key_seq):
    shuffling_list = []

    for i in range(len(block_list)):
        block_index = key_seq[i]
        shuffling_list.append(block_list[block_index])

    return shuffling_list


def mat_rotation(block, key):
    block_copy = deepcopy(block)
    block = torch.rot90(block_copy, key, [2, 3])
    return block


def mat_flipping(block, key):
    block_copy = deepcopy(block)
    block = torch.flip(block_copy, dims=[key + 1])
    return block


def sign_inversion(block, key):
    if key:
        block = block * (-1)
    return block


def channel_shuffling(block, key):
    r, g, b = torch.split(block, 1, dim=1)
    channels_order = ((r, g, b), (r, b, g), (g, r, b), (g, b, r), (b, r, g), (b, g, r))
    block = torch.cat(channels_order[key], dim=1)
    return block


def dict_remapping(model_dict):
    new_state_dict = {}
    for key, value in model_dict.items():
        if key.startswith("module."):
            new_key = key[7:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


class ModelCipher:
    def __init__(self, block_size, sub_block_size, key_sets):
        self.b_size = block_size
        self.sb_size = sub_block_size
        self.key_sets = key_sets

    # p_index->patch index
    def encryption(self, pe_weight, p_index):
        sm_list = mat_segmentation(pe_weight, self.sb_size)
        sm_list = mat_shuffling(sm_list, self.key_sets["sb_shuffling"][p_index])

        # sub-matrix
        for j in range(len(sm_list)):
            sm = sm_list[j]

            sm = mat_rotation(sm, self.key_sets["sb_rotation"][p_index][j])
            sm = mat_flipping(sm, self.key_sets["sb_flipping"][p_index][j])
            sm = sign_inversion(sm, self.key_sets["sb_NPtrans"][p_index][j])
            sm = channel_shuffling(sm, self.key_sets["sb_c_shuffling"][p_index][j])

            sm_list[j] = sm

        # pe_weight.shape
        encrypted_weight = mat_integration(sm_list, pe_weight.shape)
        return encrypted_weight


def main():
    input_size = 224
    b_size = 16
    sb_size = 8
    out_channels = 1024
    patch_nums = (input_size // b_size) ** 2

    key_sets = unpickle(f"key_set/multiple/{b_size}_{sb_size}_dict")
    model_cipher = ModelCipher(b_size, sb_size, key_sets)

    pretrained_weights_path = f"weights/baseline/convmixer_1024_20_ks9_p16_epoch125_9688.pth"
    encrypted_weights_root = f"weights/encrypted_weights/{b_size}_{sb_size}"
    if os.path.exists(encrypted_weights_root) is False:
        print("Creating saving folder..")
        os.makedirs(encrypted_weights_root)

    # model encryption
    model = timm.create_model("convmixer_1024_20_ks9_p14", pretrained=True, num_classes=10)

    if pretrained_weights_path:
        model.stem[0] = nn.Conv2d(3, out_channels, (b_size, b_size), stride=(b_size, b_size))
        pretrained_state_dict = torch.load(pretrained_weights_path)
        # key_set remapping
        pretrained_state_dict = dict_remapping(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

    for i in range(patch_nums):
        pe_weight = deepcopy(model.stem[0])
        pe_dict = pe_weight.state_dict()

        pe_dict["weight"].data = model_cipher.encryption(pe_dict["weight"], i)
        torch.save(pe_dict, f"weights/encrypted_weights/{b_size}_{sb_size}/weight4patch_{i}.pth")
        print(f"weight for patch{i} encryption completed")


if __name__ == '__main__':
    main()
