import os.path
import numpy as np
import pickle
import hashlib


def get_hash_val(img_path):
    with open(img_path, "rb") as file:
        sha256_hash = hashlib.sha256()
        while chunk := file.read(8192):
            sha256_hash.update(chunk)
        hash_value = sha256_hash.hexdigest()
        # print(hash_value)

    return hash_value


def split_hash_val(hash_value, num_keys):
    # 将哈希值按长度分割成 num_keys 份
    hash_segments = [hash_value[i:i + len(hash_value) // num_keys] for i in
                     range(0, len(hash_value), len(hash_value) // num_keys)]

    # encode the hash val as the initial state of chaos
    hash_segments = [int(h, 16) / (2 ** 16 - 1) for h in hash_segments]

    return hash_segments


def chaotic_state_update(x1, x2, x3, x4):
    a, b, c, k1, k2, k3, d = 35, 3, 35, 1, 0.2, 0.3, 5
    new_x1 = (a * (x2 - x1) + k1 * x4) % 1
    new_x2 = (c * x1 + x1 * x3 - k2 * x4) % 1
    new_x3 = ((-b) * x3 + x1 * x2 - x3 * x4) % 1
    new_x4 = (-d * x1) % 1

    return new_x1, new_x2, new_x3, new_x4


def chaotic_seq_generator(initial_state, iteration):
    init_x1, init_x2, init_x3, init_x4 = initial_state
    states_x1 = []
    states_x2 = []
    states_x3 = []
    states_x4 = []
    for i in range(iteration):
        init_x1, init_x2, init_x3, init_x4 = chaotic_state_update(init_x1, init_x2, init_x3, init_x4)
        states_x1.append(init_x1)
        states_x2.append(init_x2)
        states_x3.append(init_x3)
        states_x4.append(init_x4)
    return [states_x1, states_x2, states_x3, states_x4]


def range_linear_mapping(val, start, end):
    mapping_val = round(start + (end - start) * val)
    return mapping_val


def get_shuffled_seq(chaotic_seq, sb_nums):
    mode_list = []
    for i in range(len(chaotic_seq) // sb_nums):
        original_seq = np.arange(sb_nums)
        combined_seq = list(zip(original_seq, chaotic_seq[i:i + sb_nums]))
        combined_seq.sort(key=lambda x: x[1])
        shuffled_seq = [item[0] for item in combined_seq]
        mode_list.append(shuffled_seq)
    return mode_list


def get_rot_seq(chaotic_seq, sb_nums):
    mode_list = []

    remapping_chaotic_seq = [range_linear_mapping(e, 0, 3) for e in chaotic_seq]

    for i in range(0, len(remapping_chaotic_seq), sb_nums):
        mode = remapping_chaotic_seq[i: i + sb_nums]
        mode_list.append(mode)

    return mode_list


def get_inv_seq(chaotic_seq, sb_nums):
    mode_list = []

    remapping_chaotic_seq = [range_linear_mapping(e, 1, 2) for e in chaotic_seq]

    for i in range(0, len(remapping_chaotic_seq), sb_nums):
        mode = remapping_chaotic_seq[i: i + sb_nums]
        mode_list.append(mode)

    return mode_list


def get_np_seq(chaotic_seq, sb_nums):
    mode_list = []

    remapping_chaotic_seq = [range_linear_mapping(e, 0, 1) for e in chaotic_seq]

    for i in range(0, len(remapping_chaotic_seq), sb_nums):
        np_mode = remapping_chaotic_seq[i: i + sb_nums]
        mode_list.append(np_mode)

    return mode_list


def get_cs_seq(chaotic_seq, sb_nums):
    mode_list = []

    remapping_chaotic_seq = [range_linear_mapping(e, 0, 5) for e in chaotic_seq]

    for i in range(0, len(remapping_chaotic_seq), sb_nums):
        cs_mode = remapping_chaotic_seq[i: i + sb_nums]
        mode_list.append(cs_mode)

    return mode_list


def save_keys(key_dict, file_path):
    try:
        geeky_file = open(file_path, 'wb')
        pickle.dump(key_dict, geeky_file)
        geeky_file.close()

    except:
        print("Something went wrong")


def unpickle(file):
    with open(file, 'rb') as fo:
        img_dict = pickle.load(fo)

    return img_dict


def main():
    mb_size = 14
    sb_size = 7
    mb_num = (224 // mb_size) ** 2
    sb_nums = (mb_size // sb_size) ** 2

    key_seed = "imgs/plain_samples/Aerial.bmp"
    hash_val = get_hash_val(key_seed)
    print("hash_value:", hash_val)

    key_seed = split_hash_val(hash_val, 4)
    states = chaotic_seq_generator(key_seed, mb_num * sb_nums)

    bs_modes = get_shuffled_seq(states[0], sb_nums)
    rot_modes = get_rot_seq(states[1], sb_nums)
    inv_modes = get_inv_seq(states[1], sb_nums)
    np_modes = get_np_seq(states[2], sb_nums)
    cs_modes = get_cs_seq(states[3], sb_nums)

    # optional mode
    ms_modes = np.arange(mb_num)
    np.random.shuffle(ms_modes)

    # print("sb_shuffling:", bs_modes)
    # print("rotation:", rot_modes)
    # print("inversion:", inv_modes)
    # print("np_transformation:", np_modes)
    # print("channel_exchange:", cs_modes)

    multiple_key = {
        "sb_shuffling": bs_modes,
        "sb_rotation": rot_modes,
        "sb_flipping": inv_modes,
        "sb_NPtrans": np_modes,
        "sb_c_shuffling": cs_modes,
        "mb_shuffling": ms_modes
    }

    single_key = {
        "sb_shuffling": [bs_modes[0] for _ in range(mb_num)],
        "sb_rotation": [rot_modes[0] for _ in range(mb_num)],
        "sb_flipping": [inv_modes[0] for _ in range(mb_num)],
        "sb_NPtrans": [np_modes[0] for _ in range(mb_num)],
        "sb_c_shuffling": [cs_modes[0] for _ in range(mb_num)],
        "mb_shuffling": ms_modes
    }
    save_keys(multiple_key, os.path.join("key_set/multiple", f"{mb_size}_{sb_size}_dict"))
    save_keys(single_key, os.path.join("key_set/single", f"{mb_size}_{sb_size}_dict"))


if __name__ == '__main__':
    main()
