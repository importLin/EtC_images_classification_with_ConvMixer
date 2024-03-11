from copy import deepcopy
import torch
import numpy as np
import cv2


class ImgBlockOp:
    def block_segmentation(self, main_mat, sb_size):
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

    def block_intergration(self, block_list, mb_size):
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

    def block_shuffling(self, block_list, key_seq):
        shuffling_list = []

        for i in range(len(block_list)):
            block_index = key_seq[i]
            shuffling_list.append(block_list[block_index])

        return shuffling_list

    def get_clist(self, block_list, c_idx):
        c, h, w = block_list[0].shape
        color_list = [block_list[i][c_idx].reshape(1, h, w) for i in range(len(block_list))]

        return color_list

    def c_block_shuffling(self, block, seed):
        block = deepcopy(block)
        np.random.seed(seed)
        np.random.shuffle(block)
        return block

    def block_routing(self, block, key):
        block_copy = deepcopy(block)
        block = np.rot90(block_copy, key, axes=[1, 2])
        return block

    def block_flipping(self, block, key):
        block_copy = deepcopy(block)
        block = np.flip(block_copy, key)
        return block

    def np_transformation(self, block, key):
        if key:
            block = 255. - block

        return block

    def c_shuffling(self, block, key, e_mode):
        r, g, b = self.channel_split(block)
        if e_mode:
            opt_order = ((r, g, b), (r, b, g), (g, r, b), (g, b, r), (b, r, g), (b, g, r))
        else:
            opt_order = ((r, g, b), (r, b, g), (g, r, b),  (b, r, g), (g, b, r), (b, g, r))

        block[0, :, :], block[1, :, :], block[2, :, :] = opt_order[key]
        return block

    def channel_split(self, block):
        block_copy = deepcopy(block)
        c, h, w = block_copy.shape
        r = block_copy[0, :, :].reshape(1, h, w)
        g = block_copy[1, :, :].reshape(1, h, w)
        b = block_copy[2, :, :].reshape(1, h, w)

        return [r, g, b]


class ModelBlockOp:
    def block_segmentation(self, main_mat, sb_size):
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

    def block_intergration(self, block_list, obj_shape):
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

    def block_shuffling(self, block_list, key_seq):
        shuffling_list = []

        for i in range(len(block_list)):
            block_index = key_seq[i]
            shuffling_list.append(block_list[block_index])

        return shuffling_list

    def get_clist(self, block_list, c_idx):
        # 1024,3,7,7
        k, c, h, w = block_list[0].shape
        color_list = [block_list[i][:, c_idx, :, :].reshape(k, 1, h, w) for i in range(len(block_list))]
        return color_list

    def block_routing(self, block, key):
        block_copy = deepcopy(block)
        block = torch.rot90(block_copy, key, [2, 3])
        return block

    def block_flipping(self, block, key):
        block_copy = deepcopy(block)
        block = torch.flip(block_copy, dims=[key + 1])
        return block

    def np_transformation(self, block, key):
        if key:
            block = block * (-1)

        return block

    def channel_split(self, block):
        # c, 1, h ,w
        r, g, b = torch.split(block, 1, dim=1)
        return [r, g, b]

    def c_shuffling(self, block, key, e_mode):
        block_copy = deepcopy(block)
        r, g, b = torch.split(block_copy, 1, dim=1)
        if e_mode:
            opt_order = ((r, g, b), (r, b, g), (g, r, b), (g, b, r), (b, r, g), (b, g, r))
        else:
            opt_order = ((r, g, b), (r, b, g), (g, r, b),  (b, r, g), (g, b, r), (b, g, r))

        block = torch.cat(opt_order[key], dim=1)
        return block

