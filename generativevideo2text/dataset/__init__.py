import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


from dataset.video_dataset import gvt_video_caps_dataset, gvt_images_caps_dataset

from dataset.randaugment import RandomAugment


def create_dataset(config=None):

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    test_transform = transforms.Compose([
        transforms.Resize(
            (config['image_res'], config['image_res']), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    FRAME_DATASET = gvt_video_caps_dataset if config['use_video'] else gvt_images_caps_dataset
    test_dataset = FRAME_DATASET(
        config['test_file'],
        config['test_root'],
        max_words=config['max_length'],
        read_local_data=config['read_local_data'],
        is_train=False,
        num_frm=config['num_frm_test'],
        max_img_size=config['image_res'],
        frm_sampling_strategy='uniform',
        transform=test_transform)

    return test_dataset


def cap_collate_fn(batch):
    image_list, image_id_list = [], []
    for image, image_id in batch:
        image_list.append(image)
        image_id_list.append(image_id)
    return torch.stack(image_list, dim=0), image_id_list


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(dataset, sampler, batch_size, n_worker, collate_fn):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_worker,
        pin_memory=True,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader
