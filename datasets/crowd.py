from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
from scipy.spatial import cKDTree


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area


# 讀取圖片和人頭標註
class Crowd(data.Dataset):
    # 尋找所有 .jpg、保存裁切大小，並建立圖片正規化
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = root_path
        self.im_list = sorted(
            path for path in glob(os.path.join(self.root_path, '*.jpg'))
            if not os.path.basename(path).startswith('._')
        )
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    # 讀取指定圖片及同名 .npy 人頭座標
    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(os.path.basename(img_path).split('.')[0])
        if self.method == 'train':
            keypoints = np.load(gd_path)
            keypoints = self._ensure_nearest_distance(keypoints, gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

    @staticmethod
    def _ensure_nearest_distance(keypoints, annotation_path):
        """Ensure training annotations have (x, y, nearest-distance) columns."""
        if keypoints.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] not in (2, 3):
            raise ValueError(
                "invalid annotation shape {} in {}; expected (N, 2) or (N, 3)".format(
                    keypoints.shape, annotation_path
                )
            )
        if keypoints.shape[1] == 3:
            return keypoints

        points = keypoints.astype(np.float32, copy=False)
        point_count = len(points)
        if point_count == 1:
            nearest_distance = np.full((1, 1), 128.0, dtype=np.float32)
        else:
            neighbor_count = min(4, point_count)
            distances, _ = cKDTree(points).query(points, k=neighbor_count)
            nearest_distance = distances[:, 1:].mean(axis=1, keepdims=True).astype(np.float32)
        return np.concatenate((points, nearest_distance), axis=1)

    # 訓練資料增強
    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        # assert len(keypoints) > 0
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        re_size = random.random() * 0.5 + 0.75
        wdd = (int)(wd*re_size)
        htt = (int)(ht*re_size)
        if min(wdd, htt) >= self.c_size:
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            keypoints = keypoints*re_size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)

            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            target = np.array([])
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size

# 自動配對 IMG_x.jpg 和 hazy_IMG_x.jpg
class PairedCrowd(Crowd):
    """Return geometrically aligned clean/hazy crops for LoRA training."""

    def __init__(self, clean_root_path, hazy_root_path, crop_size,
                 downsample_ratio, is_gray=False):
        super().__init__(
            clean_root_path,
            crop_size,
            downsample_ratio,
            is_gray,
            method='train',
        )
        hazy_paths = [
            path for path in glob(os.path.join(hazy_root_path, '*.jpg'))
            if not os.path.basename(path).startswith('._')
        ]
        self.hazy_by_name = {
            self._pair_name(path): path for path in hazy_paths
        }
        missing = [
            path for path in self.im_list
            if self._pair_name(path) not in self.hazy_by_name
        ]
        if missing:
            raise FileNotFoundError(
                "missing hazy pairs for {} clean images; first missing pair: {}".format(
                    len(missing), missing[0]
                )
            )

    @staticmethod
    def _pair_name(path):
        name = os.path.splitext(os.path.basename(path))[0]
        return name[len('hazy_'):] if name.startswith('hazy_') else name

    def __getitem__(self, item):
        clean_path = self.im_list[item]
        hazy_path = self.hazy_by_name[self._pair_name(clean_path)]
        annotation_path = os.path.splitext(clean_path)[0] + '.npy'

        clean_img = Image.open(clean_path).convert('RGB')
        hazy_img = Image.open(hazy_path).convert('RGB')
        if clean_img.size != hazy_img.size:
            raise ValueError(
                "paired images have different sizes: {} {} vs {} {}".format(
                    clean_path, clean_img.size, hazy_path, hazy_img.size
                )
            )

        keypoints = np.load(annotation_path)
        keypoints = self._ensure_nearest_distance(keypoints, annotation_path)
        return self.paired_train_transform(clean_img, hazy_img, keypoints)

    def paired_train_transform(self, clean_img, hazy_img, keypoints):
        """Apply identical resize, crop, grayscale, and flip to both domains."""
        wd, ht = clean_img.size
        if random.random() > 0.88:
            clean_img = clean_img.convert('L').convert('RGB')
            hazy_img = hazy_img.convert('L').convert('RGB')

        re_size = random.random() * 0.5 + 0.75
        resized_width = int(wd * re_size)
        resized_height = int(ht * re_size)
        if min(resized_width, resized_height) >= self.c_size:
            wd = resized_width
            ht = resized_height
            clean_img = clean_img.resize((wd, ht))
            hazy_img = hazy_img.resize((wd, ht))
            keypoints = keypoints * re_size

        st_size = min(wd, ht)
        if st_size < self.c_size:
            raise ValueError(
                "image is smaller than crop size: image min side {}, crop {}".format(
                    st_size, self.c_size
                )
            )

        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        clean_img = F.crop(clean_img, i, j, h, w)
        hazy_img = F.crop(hazy_img, i, j, h, w)

        if len(keypoints) > 0:
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(inner_area / origin_area, 0.0, 1.0)
            mask = ratio >= 0.3
            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [j, i]
        else:
            target = np.array([])

        if random.random() > 0.5:
            clean_img = F.hflip(clean_img)
            hazy_img = F.hflip(hazy_img)
            if len(keypoints) > 0:
                keypoints[:, 0] = w - keypoints[:, 0]

        return (
            self.trans(clean_img),
            self.trans(hazy_img),
            torch.from_numpy(keypoints.copy()).float(),
            torch.from_numpy(target.copy()).float(),
            st_size,
        )
