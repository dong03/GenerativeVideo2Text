from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
from dataset.utils import pre_caption

decord.bridge.set_bridge("torch")


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, img):
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class dcb_video_caps_dataset(Dataset):
    def __init__(self, ann_file, video_root, max_words=30, read_local_data=True, is_train=True, num_frm=4,
                 frm_sampling_strategy="uniform", max_img_size=384, video_fmt='.mp4'
                 ):

        self.ann = open(ann_file).readlines()
        self.ann = [each.strip() for each in self.ann]

        self.max_words = max_words
        self.read_local_data = read_local_data

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(
            0.26862954, 0.26130258, 0.27577711))

    def __len__(self):
        return len(self.ann)

    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        try:
            # import pdb; pdb.set_trace()
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)

            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(
                    start_idx, end_idx, vlen / self.num_frm, dtype=int)
            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(
                    random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(
                    range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(
                    range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            else:
                raise NotImplementedError(
                    'Invalid sampling strategy {} '.format(self.frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices)
        except Exception as e:
            return np.zeros((self.num_frm, 3, self.max_img_size, self.max_img_size))

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)

        return raw_sample_frms

    def __getitem__(self, index):

        video_name = self.ann[index]
        video_path = os.path.join(
            self.video_root, f'{video_name}{self.video_fmt}')
        vid_frm_array = self._load_video_from_path_decord(
            video_path, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())

        return video, video_name


class dense_frames_caps_dataset(Dataset):
    def __init__(self, ann_file, video_root, max_words=30, read_local_data=True, is_train=True, num_frm=32, transform=None,
                 frm_sampling_strategy="uniform", max_img_size=384, video_fmt=''
                 ):

        self.ann = open(ann_file).readlines()
        self.ann = [each.strip().split('\t') for each in self.ann]

        self.max_words = max_words
        self.read_local_data = read_local_data

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(
            0.26862954, 0.26130258, 0.27577711))
        self.tranform = transform

    def __len__(self):
        return len(self.ann)

    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        try:
            # import pdb; pdb.set_trace()
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)

            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(
                    start_idx, end_idx, vlen / self.num_frm, dtype=int)
            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(
                    random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(
                    range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(
                    range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            else:
                raise NotImplementedError(
                    'Invalid sampling strategy {} '.format(self.frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices).numpy()
            # raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2).numpy()

        except Exception as e:
            raw_sample_frms = np.zeros(
                (self.num_frm, self.max_img_size, self.max_img_size, 3))

        raw_sample_frms = [Image.fromarray(each) for each in raw_sample_frms]
        return raw_sample_frms

    def __getitem__(self, index):
        try:

            video_name, cap = self.ann[index]
            video_name = video_name.split('#')[0]
            video_path = os.path.join(self.video_root, video_name + '.mp4')
            images = self._load_video_from_path_decord(video_path)
            images = [self.tranform(i) for i in images]
            images = torch.stack(images)
            cap = pre_caption(cap, 128)

        except:
            print(index, self.ann[index])
            images = torch.zeros((self.num_frm, 3, 224, 224))
            video_name = ''
            cap = ''
        return images, video_name, cap
        # import pdb; pdb.set_trace()


class dcb_images_caps_dataset(Dataset):
    def __init__(self, ann_file, video_root, max_words=30, read_local_data=True, is_train=True, num_frm=4, transform=None,
                 frm_sampling_strategy="rand", max_img_size=384, video_fmt=''
                 ):
        self.ann = []
        self.video2id = {}
        idx = 0
        for ann_file_, video_root_ in zip(ann_file, video_root):
            ann = open(ann_file_).readlines()
            ann = [each.strip().split('\t') for each in ann]
            for each in ann:
                video = each[0].split('#')[0]
                if video not in self.video2id:
                    self.video2id[video] = idx
                    idx += 1
            ann = [[os.path.join(video_root_, each[0].split('#')[
                                 0]), each[1], self.video2id[each[0].split('#')[0]]] for each in ann]
            self.ann.extend(ann)
            print(f"load {ann_file_} done")

        self.max_words = max_words
        self.read_local_data = read_local_data

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(
            0.26862954, 0.26130258, 0.27577711))
        self.tranform = transform

    def __len__(self):
        return len(self.ann)

    def _load_video_from_path(self, video_path):
        image_files = os.listdir(video_path)

        image_files = sorted(
            image_files,
            key=lambda x: int(x.split('_')[-1].split('.')[0]))

        image_files = [
            os.path.join(video_path, each) for each in image_files
        ]
        if len(image_files) != self.num_frm:
            sample_ix = np.linspace(0,
                                    len(image_files) - 1,
                                    num=self.num_frm,
                                    endpoint=True,
                                    retstep=False,
                                    dtype=int)
            image_files = np.array(image_files)[sample_ix].tolist()
        images = [Image.open(i).convert('RGB') for i in image_files]
        images = [self.tranform(i) for i in images]
        images = torch.stack(images)
        if images.shape[0] != self.num_frm:
            images = images[torch.linspace(
                0, images.shape[0]-1, self.num_frm, dtype=int)]
        return images

    def __getitem__(self, index):
        try:

            video_path, cap, video_id = self.ann[index]
            # video_name = video_name.split('#')[0]
            # video_path = os.path.join(self.video_root, video_name)
            video = self._load_video_from_path(video_path)
            cap = pre_caption(cap, 128)
        except:
            print(index)
            video = torch.zeros((self.num_frm, 3, 224, 224))
            video_path = ''
            cap = ''
            video_id = -1
        return video, video_path.split('/')[-1], cap, video_id
        # import pdb; pdb.set_trace()
