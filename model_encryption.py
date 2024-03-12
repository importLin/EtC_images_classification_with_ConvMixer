import torch
import timm
import os

from key_generator import unpickle
from block_operation import ModelBlockOp
from copy import deepcopy
from torch import nn


def dict_remapping(model_dict):
    new_state_dict = {}
    for key, value in model_dict.items():
        if key.startswith("module."):
            new_key = key[7:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


class ModelCipher(ModelBlockOp):
    def __init__(self, block_size, sub_block_size, key_sets):
        self.b_size = block_size
        self.sb_size = sub_block_size
        self.key_sets = key_sets

    # p_index->patch index
    def encryption(self, pe_weight, p_index):
        sm_list = self.block_segmentation(pe_weight, self.sb_size)
        sm_list = self.block_shuffling(sm_list, self.key_sets["sb_shuffling"][p_index])

        # sub-matrix
        for j in range(len(sm_list)):
            sm = sm_list[j]

            sm = self.block_routing(sm, self.key_sets["sb_rotation"][p_index][j])
            sm = self.block_flipping(sm, self.key_sets["sb_flipping"][p_index][j])
            sm = self.np_transformation(sm, self.key_sets["sb_NPtrans"][p_index][j])
            sm = self.c_shuffling(sm, self.key_sets["sb_c_shuffling"][p_index][j])

            sm_list[j] = sm

        # pe_weight.shape
        encrypted_weight = self.block_intergration(sm_list, pe_weight.shape)
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


if __name__ == '__main__':
    main()
