import os.path
import cv2
import os
from block_operation import ImgBlockOp
from key_generator import unpickle


class ImgGenerator(ImgBlockOp):
    def __init__(self, img_size, block_size, sub_block_size, key_sets):
        self.img_size = img_size
        self.b_size = block_size
        self.sb_size = sub_block_size
        self.key_sets = key_sets

    def encryption(self, plain_img):
        plain_img = cv2.resize(plain_img, (self.img_size, self.img_size))
        rgb_img = cv2.cvtColor(plain_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

        b_list = self.block_segmentation(rgb_img, self.b_size)
        for i in range(len(b_list)):
            b = b_list[i]
            sb_list = self.block_segmentation(b, self.sb_size)
            sb_list = self.block_shuffling(sb_list, self.key_sets["sb_shuffling"][i])
            for j in range(len(sb_list)):
                sb = sb_list[j]
                sb = self.block_routing(sb, self.key_sets["sb_rotation"][i][j])
                sb = self.block_flipping(sb, self.key_sets["sb_flipping"][i][j])
                sb = self.np_transformation(sb, self.key_sets["sb_NPtrans"][i][j])
                sb = self.c_shuffling(sb, self.key_sets["sb_c_shuffling"][i][j], 1)

                sb_list[j] = sb

            b_list[i] = self.block_intergration(sb_list, self.b_size)
        # b_list = self.block_shuffling(b_list, self.key_sets["mb_shuffling"])

        encrypted_img = self.block_intergration(b_list, self.img_size).transpose(1, 2, 0)
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
        os.majikedirs(saving_root)

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
