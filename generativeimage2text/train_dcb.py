from .common import Config
import json
import os.path as op
from .common import qd_tqdm as tqdm
from .common import json_dump
from .common import pilimg_from_base64
from .torch_common import recursive_to_device
from .tsv_io import TSVFile, tsv_writer, tsv_reader
from .common import write_to_file
import torch
from .torch_common import torch_load
import PIL
import numpy as np
from pprint import pformat
import logging
from transformers import BertTokenizer
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from azfuse import File

from .common import init_logging
from .common import parse_general_args
from .tsv_io import load_from_yaml_file
from .torch_common import torch_load
from .torch_common import load_state_dict
from .torch_common import resize_2d_pos_embed
from .layers.CLIP import clip
from .layers.decoder import (TransformerDecoderTextualHead,
                             AutoRegressiveBeamSearch, GeneratorWithBeamSearch)
from .layers.decoder import CaptioningModel
from .process_image import load_image_by_pil
from .data_layer.transform import RenameKey, SelectTransform
from .data_layer.transform import ImageTransform2Dict
from .data_layer.transform import get_inception_train_transform
from .data_layer.builder import collate_fn
from .model import get_git_model


def get_data(image_file, prefix, target, tokenizer, image_transform):
    max_text_len = 40
    prefix_encoding = tokenizer(
        prefix, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    target_encoding = tokenizer(
        target, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
    payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
    if len(payload) > max_text_len:
        payload = payload[-(max_text_len - 2):]
        need_predict = need_predict[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload + [tokenizer.sep_token_id]
    need_predict = [0] + need_predict + [1]
    if isinstance(image_file, str):
        im = load_image_by_pil(image_file)
    elif isinstance(image_file, list):
        im = [load_image_by_pil(image_file_) for image_file_ in image_file]

    data = {
        'caption_tokens': torch.tensor(input_ids),
        #'caption_lengths': len(input_ids),
        'need_predict': torch.tensor(need_predict),
        'image': im,
        # 'rect' field can be fed in 'caption', which tells the bounding box
        # region of the image that is described by the caption. In this case,
        # we can optionally crop the region.
        'caption': {},
        # this iteration can be used for crop-size selection so that all GPUs
        # can process the image with the same input size
        'iteration': 0,
    }
    data = image_transform(data)

    return data

def get_image_transform(cfg):
    return get_multi_scale_image_transform(cfg, is_train=True)

def get_default_mean():
    return [0.485, 0.456, 0.406]

def get_default_std():
    return [0.229, 0.224, 0.225]

def get_transform_image_norm(cfg, default=None):
    if cfg.data_normalize == 'default':
        normalize = transforms.Normalize(
            mean=get_default_mean(), std=get_default_std())
    elif cfg.data_normalize == 'clip':
        # clip model
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    else:
        raise NotImplementedError(cfg.data_normalize)
    return normalize

def get_transform_vit_default(cfg, is_train):
    default_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = get_transform_image_norm(cfg, default_normalize)
    transform = get_inception_train_transform(
        bgr2rgb=True,
        crop_size=cfg.train_crop_size,
        normalize=normalize,
        small_scale=cfg.input_small_scale,
        no_color_jitter=cfg.no_color_jitter,
        no_flip=cfg.no_flip,
        no_aspect_dist=cfg.no_aspect_dist,
        resize_crop=cfg.resize_crop,
        max_size=cfg.train_max_size,
        interpolation=cfg.interpolation or Image.BILINEAR,
    )
    return transform

def get_transform_image(cfg, is_train):
    train_transform = cfg.train_transform
    if train_transform == 'vitp':
        transform = get_transform_vit_default(
            cfg, is_train=is_train)
    else:
        raise NotImplementedError(train_transform)
    return transform

class ImageTransform2Images(object):
    def __init__(self, sep_transform, first_joint=None):
        self.image_transform = sep_transform
        self.first_joint = first_joint

    def __call__(self, imgs):
        if self.first_joint is not None:
            imgs = self.first_joint(imgs)
        return [self.image_transform(im) for im in imgs]

    def __repr__(self):
        return 'ImageTransform2Images(image_transform={})'.format(
            self.image_transform,
        )

def get_transform_images(cfg, is_train):
    trans = get_transform_image(cfg, is_train)
    trans = ImageTransform2Images(trans)
    return trans

def trans_select_for_crop_size(
    data, train_crop_sizes,
    iteration_multi=0,
):
    if iteration_multi <= 0:
        if len(train_crop_sizes) == 1:
            idx = 0
        else:
            idx = data['iteration'] % len(train_crop_sizes)
    elif data['iteration'] <= iteration_multi:
        idx = data['iteration'] % len(train_crop_sizes)
    else:
        idx = -1
    return idx

def get_multi_scale_image_transform(cfg, is_train, get_one=get_transform_image):
    def get_multi_res_transform(s):
        old = cfg.train_crop_size if is_train else cfg.test_crop_size
        all_t = []
        multi_res_factors = cfg.multi_res_factors or []
        for i, f in enumerate(multi_res_factors):
            if is_train:
                cfg.train_crop_size = s // f
            else:
                cfg.test_crop_size = s // f
            key = 'image_{}'.format(i)
            all_t.append(RenameKey({'image': key}, not_delete_origin=True))
            t = get_one(cfg, is_train)
            t = ImageTransform2Dict(t, key=key)
            all_t.append(t)
        # get_one depends on train_crop_size
        if is_train:
            cfg.train_crop_size = s
        else:
            cfg.test_crop_size = s
        t = get_one(cfg, is_train)
        t = ImageTransform2Dict(t)
        all_t.append(t)
        if is_train:
            cfg.train_crop_size = old
        else:
            cfg.test_crop_size = old
        return transforms.Compose(all_t)

    if is_train:
        if cfg.min_size_range32 is None:
            train_crop_sizes = [cfg.train_crop_size]
        else:
            train_crop_sizes = list(range(
                cfg.min_size_range32[0],
                cfg.min_size_range32[1] + cfg.patch_size - 1, cfg.patch_size,
            ))
    else:
        train_crop_sizes = [cfg.test_crop_size]

    crop_trans = []
    for s in train_crop_sizes:
        t = get_multi_res_transform(s)
        crop_trans.append(t)
    iteration_multi = 0
    image_transform = SelectTransform(
        crop_trans,
        lambda d: trans_select_for_crop_size(
            d, train_crop_sizes, iteration_multi))
    return image_transform

def forward_backward_example(image_files, captions, prefixs=None):
    if prefixs is None:
        prefixs = [''] * len(captions)
    cfg = {
        'crop_region_extend_in_datatransform': 4,
        'data_normalize': 'clip',
        'train_crop_size': 224,
        'input_small_scale': 0.8,
        'no_color_jitter': True,
        'no_flip': True,
        'no_aspect_dist': True,
        'interpolation': 'bicubic',
        'min_size_range32': [160, 224], # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
        'patch_size': 16,
        'train_transform': 'vitp',
    }
    cfg = Config(cfg, {})
    all_data = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    image_transform = get_image_transform(cfg)
    for image_file, prefix, target in zip(image_files, prefixs, captions):
        data = get_data(image_file, prefix, target,
                        tokenizer, image_transform)
        all_data.append(data)
    data = collate_fn(all_data)
    logging.info(image_transform)
    data = recursive_to_device(data, 'cuda')

    param = {}
    model = get_git_model(tokenizer, param)
    model.train()
    model.cuda()
    loss_dict = model(data)
    loss = sum(loss_dict.values())
    loss.backward()
    logging.info(loss)


def dcb_train():
    import os
    import random
    from torch.cuda.amp import autocast, GradScaler
    from transformers import ChineseCLIPProcessor
    from .common import Progbar
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings('ignore')
    EPOCH = 10
    STOP_EPOCH = 5
    FAIL_EPOCH = 0
    BZ = 4
    INIT_LR = 1.0e-5
    print_freq = 50
    # ==================dataset==================
    root = '/data/dcb/bv/FrameWithTextData'
    bvcaptions = open(
        '/home/dcb/code/bv/captioning/merge_b5w_goodcaptions.txt',
        encoding='utf-8').readlines()
    bvcaptions = [each.strip().split('\t') for each in bvcaptions]
    select_bv = []
    for line in bvcaptions:
        bv = line[0]
        if len(os.listdir(os.path.join(root, bv))):
            select_bv.append(line)
    print(len(bvcaptions), len(select_bv))
    bvcaptions = select_bv
    train_raw_data, val_raw_data = [], []
    import random
    random.seed(42)
    random.shuffle(bvcaptions)
    for line in bvcaptions[:-500]:
        bv, gts = line[0], line[1:]
        train_raw_data.extend([[os.path.join(root, bv), gt] for gt in gts])

    for line in bvcaptions[-500:]:
        bv, gts = line[0], line[1:]
        val_raw_data.extend([[os.path.join(root, bv), gt] for gt in gts])

    # ==================model==================
    cfg = {
        'crop_region_extend_in_datatransform': 4,
        'data_normalize': 'clip',
        'train_crop_size': 224,
        'input_small_scale': 0.8,
        'no_color_jitter': True,
        'no_flip': True,
        'no_aspect_dist': True,
        'interpolation': 'bicubic',
        'min_size_range32': [
            160, 224
        ],  # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
        'patch_size': 16,
        'train_transform': 'vitp',
    }
    cfg = Config(cfg, {})
    param = {}
    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(
                f'aux_data/models/{model_name}/parameter.yaml')
    model_name = 'GIT_BASE_VATEX'
    tokenizer = ChineseCLIPProcessor.from_pretrained(
        "OFA-Sys/chinese-clip-vit-base-patch16")
    tokenizer = tokenizer.tokenizer
    image_transform = get_image_transform(cfg)
    model = get_git_model(tokenizer, param)
    pretrained = f'output/{model_name}/snapshot/model.pt'
    checkpoint = torch_load(pretrained)['model']
    load_state_dict(model, checkpoint)
    model.cuda()
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           EPOCH,
                                                           eta_min=0,
                                                           last_epoch=-1,
                                                           verbose=False)

    best_score = 1e5
    best_epoch = -1
    for epoch in range(EPOCH):
        # train
        # '''
        model.train()
        header = 'Train Epoch: [{}]'.format(epoch)
        random.shuffle(train_raw_data)
        progbar = Progbar(len(train_raw_data) // BZ)
        for data_ix in range(0, len(train_raw_data), BZ):
            select_data = train_raw_data[data_ix:data_ix + BZ]
            input_data = []
            for image_path, captions in select_data:
                image_files = os.listdir(image_path)

                image_files = sorted(
                    image_files,
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
                image_files = [
                    os.path.join(image_path, each) for each in image_files
                ]

                data = get_data(image_files, '', captions, tokenizer,
                                image_transform)
                input_data.append(data)
            input_data = collate_fn(input_data)
            try:
                optimizer.zero_grad()
                input_data = recursive_to_device(input_data, 'cuda')
                with autocast():
                    loss_dict = model(input_data)
                    loss = sum(loss_dict.values())
                progbar.add(1, values=[
                    ('train_loss', loss.item()),
                ])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            except:
                print(select_data)
            # scaler, optimizer, logger
        scheduler.step()
        torch.save(model.state_dict(), f'/data4/dcb/records/git/ckpt/GIT_DCB_epoch{epoch}.pth')
        # '''
        # val
        model.eval()
        loss_score = []
        header = 'Val Epoch: [{}]'.format(epoch)
        progbar = Progbar(len(val_raw_data) // BZ)
        # cal_loss
        model.training = True
        for data_ix in range(0, len(val_raw_data), BZ):
            select_data = val_raw_data[data_ix:data_ix + BZ]
            input_data = []
            for image_path, captions in select_data:
                image_files = os.listdir(image_path)
                image_files = sorted(
                    image_files,
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
                image_files = [
                    os.path.join(image_path, each) for each in image_files
                ]
                data = get_data(image_files, '', captions, tokenizer,
                                image_transform)
                input_data.append(data)

            input_data = collate_fn(input_data)

            input_data = recursive_to_device(input_data, 'cuda')
            with torch.no_grad():
                loss_dict = model(input_data)
                loss = sum(loss_dict.values())
                loss_score.append(loss.item())
            progbar.add(1, values=[
                ('val_loss', loss.item()),
            ])
            
        score = np.mean(loss_score)
        if score <= best_score:
            FAIL_EPOCH = 0
            best_score = score
            best_epoch = epoch
        else:
            FAIL_EPOCH += 1

        if FAIL_EPOCH == STOP_EPOCH:
            print(
                f"Early stop at epoch {epoch}, get best in epoch {best_epoch}")
            break

        # cal decode result
        '''
        f = open(f'ckpt/val_res_epo{epoch}.txt','w', encoding='utf-8')
        progbar = Progbar(len(val_raw_data) // BZ)
        # cal_loss
        model.training = False
        for data_ix in range(0, len(val_raw_data), BZ):
            select_data = val_raw_data[data_ix:data_ix + BZ]
            input_data = []
            for image_path, captions in select_data:
                image_files = os.listdir(image_path)
                image_files = sorted(
                    image_files,
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
                image_files = [
                    os.path.join(image_path, each) for each in image_files
                ]
                data = get_data(image_files, '', captions, tokenizer,
                                image_transform)
                input_data.append(data)

            input_data = collate_fn(input_data)
            
            input_data = recursive_to_device(input_data, 'cuda')
            with torch.no_grad():
                import pdb; pdb.set_trace()
                result = model(input_data)
                
                for i in range(result['predictions'].shape[0]):
                    cap = tokenizer.decode(
                            result['predictions'][0],
                            skip_special_tokens=True)
                    f.write(f"{image_path}\t{cap}\n")

            progbar.add(1, values=[
                ('val_loss', loss.item()),
            ])
        f.close()
        '''
def speed_test_forward_backward():
    duplicate = 32
    image_files = ['aux_data/images/1.jpg', 'aux_data/images/2.jpg'] * duplicate
    captions = ['a couple of boats in a large body of water.', 'a view of a mountain with a tree'] * duplicate

    prefixs = [''] * len(captions)
    cfg = {
        'crop_region_extend_in_datatransform': 4,
        'data_normalize': 'clip',
        'train_crop_size': 224,
        'input_small_scale': 0.8,
        'no_color_jitter': True,
        'no_flip': True,
        'no_aspect_dist': True,
        'interpolation': 'bicubic',
        'min_size_range32': [160, 224], # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
        'patch_size': 16,
        'train_transform': 'vitp',
    }
    cfg = Config(cfg, {})
    all_data = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    image_transform = get_image_transform(cfg)
    for image_file, prefix, target in zip(image_files, prefixs, captions):
        data = get_data([image_file, image_file, image_file, image_file],
                        prefix, target, tokenizer, image_transform)
        all_data.append(data)
        # break
    data = collate_fn(all_data)
    logging.info(image_transform)
    data = recursive_to_device(data, 'cuda')
    data['image'] = data['image'].to(torch.float16)
    param = {}
    model = get_git_model(tokenizer, param)
    model.train()
    model.cuda()
    model.half()

    # warmup
    for _ in range(2):
        import pdb
        pdb.set_trace()
        loss_dict = model(data)
        loss = sum(loss_dict.values())
        loss.backward()

    import time
    start = time.time()
    for iteration in range(1000):
        loss_dict = model(data)
        loss = sum(loss_dict.values())
        loss.backward()
        if (iteration % 10) == 0:
            end = time.time()
            speed = data['image'].shape[0] * 100 / (end - start)
            if iteration > 0:
                logging.info('speed = {}'.format(speed))
            start = time.time()

    logging.info(loss)


if __name__ == '__main__':
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

