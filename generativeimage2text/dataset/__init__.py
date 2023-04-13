import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


from dataset.video_dataset_dcb import dcb_video_caps_dataset, dcb_images_caps_dataset, dense_frames_caps_dataset

from dataset.randaugment import RandomAugment


def create_dataset(dataset, config, epoch=None):

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(
            0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(
            (config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'dcb_video_caps':
        test_dataset = dcb_video_caps_dataset(config['test_file'], config['test_root'], max_words=config['max_length'], read_local_data=config['read_local_data'],
                                              is_train=False, num_frm=config['num_frm_test'], max_img_size=config['image_res'], frm_sampling_strategy='uniform')
        return test_dataset
    elif dataset == 'dcb_frames_caps':
        FRAME_DATASET = dense_frames_caps_dataset if config['dense'] else dcb_images_caps_dataset
        train_dataset = FRAME_DATASET(config['train_file'],
                                      config['train_root'],
                                      max_words=config['max_length'],
                                      read_local_data=config['read_local_data'],
                                      is_train=True,
                                      num_frm=config['num_frm_test'],
                                      max_img_size=config['image_res'],
                                      frm_sampling_strategy='uniform',
                                      transform=train_transform)

        val_dataset = FRAME_DATASET(config['val_file'], config['val_root'], max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, num_frm=config['num_frm_test'], max_img_size=config['image_res'], frm_sampling_strategy='uniform', transform=test_transform)
        test_dataset = FRAME_DATASET(config['test_file'], config['test_root'], max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=False, num_frm=config['num_frm_test'], max_img_size=config['image_res'], frm_sampling_strategy='uniform', transform=test_transform)
        return train_dataset, val_dataset, test_dataset


def cap_collate_fn(batch):
    image_list, image_id_list, caps = [], [], []
    for image, image_id, cap in batch:
        image_list.append(image)
        image_id_list.append(image_id)
        caps.append(cap)
    return torch.stack(image_list, dim=0), image_id_list, caps


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
