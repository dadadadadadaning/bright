import random
import numpy as np
import cv2

def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img

def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label



def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label


def random_scale(pre_img, post_img, label=None, scales=None):
    scale = random.choice(scales)
    if scales != 1:
        sh = int(pre_img.shape[0] * scale)
        sw = int(pre_img.shape[1] * scale)
        pre_img = cv2.resize(pre_img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        post_img = cv2.resize(post_img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if label is not None:
        label = cv2.resize(label, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return pre_img, post_img, label

def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label


def random_crop(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
   
    return pre_img, post_img, label


def damage_aware_random_crop(
    pre_img,
    post_img,
    label,
    crop_size,
    damage_crop_prob=0.5,
    intact_crop_prob=0.25,
    random_crop_prob=0.25,
    min_damage_pixels=20,
    max_crop_retry=20,
    mean_rgb=[0, 0, 0],
    ignore_index=255,
):
    """Random crop biased toward damaged/destroyed or intact pixels when requested."""
    h, w = label.shape
    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index
    for c in range(3):
        pad_pre_image[:, :, c] = mean_rgb[c]
        pad_post_image[:, :, c] = mean_rgb[c]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))
    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    total_prob = float(damage_crop_prob) + float(intact_crop_prob) + float(random_crop_prob)
    if total_prob <= 0:
        mode = "random"
    else:
        r = random.random() * total_prob
        if r < float(damage_crop_prob):
            mode = "damage"
        elif r < float(damage_crop_prob) + float(intact_crop_prob):
            mode = "intact"
        else:
            mode = "random"

    def random_box():
        hs = random.randrange(0, H - crop_size + 1, 1)
        ws = random.randrange(0, W - crop_size + 1, 1)
        return hs, hs + crop_size, ws, ws + crop_size

    def box_around_pixel(y, x):
        h_min = max(0, y - crop_size + 1)
        h_max = min(y, H - crop_size)
        w_min = max(0, x - crop_size + 1)
        w_max = min(x, W - crop_size)
        hs = random.randint(int(h_min), int(h_max)) if h_max >= h_min else 0
        ws = random.randint(int(w_min), int(w_max)) if w_max >= w_min else 0
        return hs, hs + crop_size, ws, ws + crop_size

    def target_box(target_mode):
        if target_mode == "damage":
            coords = np.argwhere((pad_label == 2) | (pad_label == 3))
            min_pixels = int(min_damage_pixels)
            verify = lambda crop: int(((crop == 2) | (crop == 3)).sum()) >= min_pixels
        elif target_mode == "intact":
            coords = np.argwhere(pad_label == 1)
            verify = lambda crop: int((crop == 1).sum()) > 0
        else:
            return random_box()

        if coords.size == 0:
            return None
        for _ in range(int(max_crop_retry)):
            y, x = coords[random.randrange(0, len(coords))]
            hs, he, ws, we = box_around_pixel(int(y), int(x))
            if verify(pad_label[hs:he, ws:we]):
                return hs, he, ws, we
        return None

    box = target_box(mode)
    if box is None and mode == "damage":
        box = target_box("intact")
    if box is None:
        box = random_box()

    H_start, H_end, W_start, W_end = box
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
    return pre_img, post_img, label
