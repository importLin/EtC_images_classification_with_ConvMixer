import torch.nn.functional as F
import torch
import timm
import os
import cv2

from torch import nn
from torchvision import transforms as T

from copy import deepcopy

def prediction(img_tensor, model, device):
    model = model.eval().to(device)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    with torch.no_grad():
        out = model(img_tensor)

    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 3)

    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

    return out


def loading_img(img_path, transformer):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = transformer(img).unsqueeze(0)
    img_tensor = img_tensor
    return img_tensor


class DynaMicConv(nn.Module):
    def __init__(self, out_channels, weight_list, bias_list, img_size, patch_size):
        super(DynaMicConv, self).__init__()
        self.patch_nums = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.img_size = img_size
        self.out_chennels = out_channels

        self.weight_list = nn.ParameterList(
            [nn.Parameter(w) for _, w in enumerate(weight_list)]
        )
        self.bias_list = nn.ParameterList(
            [nn.Parameter(b) for _, b in enumerate(bias_list)]
        )

    def forward(self, x):
        # b,c,h,w
        out_shape = (-1, self.out_chennels, self.img_size // self.patch_size, self.img_size // self.patch_size)
        out_list = []
        p_idx = 0
        for i in range(self.img_size // self.patch_size):
            for j in range(self.img_size // self.patch_size):
                conv_w = self.weight_list[p_idx]
                conv_b = self.bias_list[p_idx]

                patch = x[:, :, self.patch_size * i: self.patch_size + (self.patch_size * i),
                        self.patch_size * j: self.patch_size + (self.patch_size * j)]
                out = F.conv2d(patch, conv_w, conv_b,
                               stride=self.patch_size, padding=0)

                out_list.append(out)
                p_idx += 1

        final_out = torch.cat(out_list, dim=2).reshape(out_shape)
        return final_out


def conv_initialize(out_channels, img_size, patch_size, weights_root):
    weight_list = []
    bias_list = []
    for i in range((img_size//patch_size)**2):
        conv_dict = torch.load(os.path.join(weights_root, f"weight4patch_{i}.pth"))
        weight_list.append(conv_dict["weight"].data)
        bias_list.append(conv_dict["bias"].data)

    return DynaMicConv(out_channels, weight_list, bias_list, img_size, patch_size)


def main():
    device = "cuda:0"
    img_size = 224
    patch_size = 14
    sb_size = 7
    out_channels = 1024
    weights_root = f"weights/encrypted_weights/{patch_size}_{sb_size}"

    ori_model = timm.create_model("convmixer_1024_20_ks9_p14", pretrained=True)
    ind_model = timm.create_model("convmixer_1024_20_ks9_p14", pretrained=True)
    ind_model.stem[0] = conv_initialize(out_channels, img_size, patch_size, weights_root)

    transformer = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000]),
                    std=torch.tensor([0.5000, 0.5000, 0.5000]))
    ])
    ori_x = loading_img("imgs/plain_samples/lena.png", transformer).to(device)
    ind_x = loading_img(f"imgs/etc_samples/{patch_size}_{sb_size}/ind_{patch_size}_{sb_size}_lena.png",
                        transformer).to(device)

    print("original model output:")
    ori_out = prediction(ori_x, ori_model, device)
    print("\nencrypted model output:")
    ind_out = prediction(ind_x, ind_model, device)
    print("difference = ", torch.sum(ori_out - ind_out).item())


if __name__ == '__main__':
    main()
