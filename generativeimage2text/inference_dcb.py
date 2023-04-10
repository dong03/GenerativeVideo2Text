import json
import os
from torch.cuda.amp import autocast
import numpy as np
import os.path as op
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .common import qd_tqdm as tqdm
from .common import json_dump, Config
from .common import pilimg_from_base64
from .common import get_mpi_rank, get_mpi_size, get_mpi_local_rank

from .tsv_io import TSVFile, tsv_writer, tsv_reader
from .common import write_to_file
import torch
import PIL
from pprint import pformat
import logging
from transformers import BertTokenizer, ChineseCLIPProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from azfuse import File
from .train import get_transform_image
from .common import init_logging
from .common import parse_general_args
from .tsv_io import load_from_yaml_file
from .torch_common import torch_load
from .torch_common import load_state_dict
from .process_image import load_image_by_pil
from .model import get_git_model
from transformers import ChineseCLIPProcessor


class MinMaxResizeForTest(object):

    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size

        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __repr__(self):
        return 'MinMaxResizeForTest({}, {})'.format(self.min_size,
                                                    self.max_size)

    def __call__(self, img):
        size = self.get_size(img.size)
        import torchvision.transforms.functional as F
        image = F.resize(img, size, interpolation=PIL.Image.BICUBIC)
        return image


def download_model():
    model_names = [
        'GIT_BASE', 'GIT_BASE_COCO', 'GIT_BASE_TEXTCAPS', 'GIT_BASE_VATEX'
    ]
    for model_name in model_names:
        param = {}
        if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
            param = load_from_yaml_file(
                f'aux_data/models/{model_name}/parameter.yaml')

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                  do_lower_case=True)
        # model
        # import pdb; pdb.set_trace()
        model = get_git_model(tokenizer, param)
        pretrained = f'output/{model_name}/snapshot/model.pt'
        # checkpoint = torch.load(pretrained, map_location='cpu')
        # tobeload = {}
        # for k, v in checkpoint.items():
        #     tobeload[k.replace('git.','')] = v
        checkpoint = torch_load(pretrained)['model']
        load_state_dict(model, checkpoint)
        model.eval()
        print(f"load {model_name} sucess!")


def batch_dcb_inference_single_image(model_name, prefix, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")

    param = {}
    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(
            f'aux_data/models/{model_name}/parameter.yaml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    transforms = get_image_transform(param)

    # model
    model = get_git_model(tokenizer, param)
    pretrained = f'output/{model_name}/snapshot/model.pt'
    # import pdb; pdb.set_trace()
    checkpoint = torch.load(pretrained, map_location='cpu')['model']
    load_state_dict(model, checkpoint)
    model.to(device)
    model.eval()

    # prefix
    max_text_len = 40
    prefix_encoding = tokenizer(prefix,
                                padding='do_not_pad',
                                truncation=True,
                                add_special_tokens=False,
                                max_length=max_text_len)
    payload = prefix_encoding['input_ids']
    if len(payload) > max_text_len - 2:
        payload = payload[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload

    root = '/data/dcb/bv/FrameWithTextData'
    prefix = torch.tensor(input_ids).unsqueeze(0).to(device)
    res = []
    videos = open(
        '/data/dcb/bv/all_video_group_withtext_withDesc_keep_infos.txt',
        encoding='utf-8').readlines()
    videos = [each.strip().split('\t')[0] for each in videos]
    with open(f'{model_name}.txt', 'w') as f:
        for ix, bv in enumerate(tqdm(videos)):
            try:
                # import pdb; pdb.set_trace()
                import time
                st = time.time()
                image_path = os.listdir(f'{root}/{bv}')
                if not len(image_path):
                    continue

                image_path = sorted(
                    image_path,
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
                if len(image_path) > 8:
                    sample_ix = np.linspace(0,
                                            len(image_path) - 1,
                                            num=8,
                                            endpoint=True,
                                            retstep=False,
                                            dtype=int)
                    image_path = np.array(image_path)[sample_ix]
                img = [
                    load_image_by_pil(f'{root}/{bv}/{i}') for i in image_path
                ]
                img = [transforms(i) for i in img]
                img = [i.unsqueeze(0).to(device) for i in img]
                # print("load image:", time.time() - st)
                st = time.time()
                # if ix == 7:
                #     import pdb; pdb.set_trace()
                with torch.no_grad():
                    result = model({
                        'image': img,
                        'prefix': prefix,
                    })
                # print("inference:", time.time() - st)
                st = time.time()
                cap = tokenizer.decode(result['predictions'][0].tolist(),
                                       skip_special_tokens=True)
                # res.append(f"{bv}\t{cap}\n")
                f.write(f"{bv}\t{cap}\n")
                # print("decode:", time.time() - st)
            except:
                print(bv)


def each_dcb_inference_single_image(model_name, prefix, gpu_id, ckpt=None):
    device = torch.device(f"cuda:{gpu_id}")

    param = {}
    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(
            f'aux_data/models/{model_name}/parameter.yaml')

    if ckpt:
        tokenizer = ChineseCLIPProcessor.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16")
        tokenizer = tokenizer.tokenizer
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                  do_lower_case=True)
    transforms = get_image_transform(param)

    # model
    model = get_git_model(tokenizer, param)

    # cfg = {
    #     'crop_region_extend_in_datatransform': 4,
    #     'data_normalize': 'clip',
    #     'train_crop_size': 224,
    #     'input_small_scale': 0.8,
    #     'no_color_jitter': True,
    #     'no_flip': True,
    #     'no_aspect_dist': True,
    #     'interpolation': 'bicubic',
    #     'min_size_range32': [
    #         160, 224
    #     ],  # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
    #     'patch_size': 16,
    #     'train_transform': 'vitp',
    # }
    # cfg = Config(cfg, {})
    # param = {}

    # tokenizer = ChineseCLIPProcessor.from_pretrained(
    #     "OFA-Sys/chinese-clip-vit-base-patch16")
    # tokenizer = tokenizer.tokenizer
    # transforms = get_transform_image(cfg, False)
    # model = get_git_model(tokenizer, param)
    if ckpt:
        pass
        checkpoint = torch.load(ckpt, map_location='cpu')['model']
        load_state_dict(model, checkpoint)
    else:
        pretrained = f'output/{model_name}/snapshot/model.pt'
        # import pdb; pdb.set_trace()
        checkpoint = torch.load(pretrained, map_location='cpu')['model']
        # ckpt = torch.load('ckpt/GIT_BV_epoch9.pth', map_location='cpu')
        # model.load_state_dict(ckpt)
        load_state_dict(model, checkpoint)
    model.to(device)
    model.eval()

    # prefix
    max_text_len = 40
    prefix_encoding = tokenizer(prefix,
                                padding='do_not_pad',
                                truncation=True,
                                add_special_tokens=False,
                                max_length=max_text_len)
    payload = prefix_encoding['input_ids']
    if len(payload) > max_text_len - 2:
        payload = payload[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload
    prefix = torch.tensor(input_ids).unsqueeze(0).to(device)
    root = '/data/dcb/bv/FrameWithTextData'
    file = '/home/dcb/code/bv/git_aimc/BV_0321_caption.txt'
    # file = '/home/dcb/code/bv/git_aimc/bvcap_test_cap.txt'
    videos = open(file, encoding='utf-8').readlines()
    videos = [each.strip().split('\t')[0] for each in videos]
    name_plus = os.path.split(ckpt)[-1] if ckpt else ''
    with open(f'{model_name}_{name_plus}_cmo.txt', 'w') as f:
        for ix, bv in enumerate(tqdm(videos)):
            try:
                # import pdb; pdb.set_trace()
                import time
                st = time.time()
                image_path = os.listdir(f'{root}/{bv}')
                if not len(image_path):
                    continue

                image_path = sorted(
                    image_path,
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
                if len(image_path) > 4:
                    sample_ix = np.linspace(0,
                                            len(image_path) - 1,
                                            num=4,
                                            endpoint=True,
                                            retstep=False,
                                            dtype=int)
                    image_path = np.array(image_path)[sample_ix]
                img = [
                    load_image_by_pil(f'{root}/{bv}/{i}') for i in image_path
                ]
                img = [transforms(i) for i in img]
                img = [i.unsqueeze(0).to(device) for i in img]
                # print("load image:", time.time() - st)
                st = time.time()
                # if ix == 7:
                #     import pdb; pdb.set_trace()
                ress = []
                with torch.no_grad():
                    result = model({
                        'image': img,
                        'prefix': prefix,
                    })
                    ress = tokenizer.decode(result['predictions'][0].tolist(),
                                            skip_special_tokens=True)
                    # import pdb; pdb.set_trace()
                    # for i_ in img:

                    #     result = model({
                    #         'image': [i_],
                    #         'prefix': prefix,
                    #     })

                    #     # print("inference:", time.time() - st)
                    #     st = time.time()
                    #     cap = tokenizer.decode(
                    #         result['predictions'][0].tolist(),
                    #         skip_special_tokens=True)

                    #     # res.append(f"{bv}\t{cap}\n")
                    #     ress.append(cap)
                # import pdb; pdb.set_trace()
                # ress = "\t".join(ress)
                f.write(f"{bv}\t{ress}\n")
                # print("decode:", time.time() - st)
            except:
                print(bv)


def test_git_inference_single_image(image_path, model_name, prefix):
    param = {}
    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(
            f'aux_data/models/{model_name}/parameter.yaml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    if isinstance(image_path, str):
        image_path = [image_path]
    # if it is more than 1 image, it is normally a video with multiple image
    # frames
    img = [load_image_by_pil(i) for i in image_path]

    transforms = get_image_transform(param)
    img = [transforms(i) for i in img]

    # model
    model = get_git_model(tokenizer, param)
    pretrained = f'output/{model_name}/snapshot/model.pt'
    checkpoint = torch_load(pretrained)['model']
    load_state_dict(model, checkpoint)
    model.cuda()
    model.eval()
    img = [i.unsqueeze(0).cuda() for i in img]

    # prefix
    max_text_len = 40
    prefix_encoding = tokenizer(prefix,
                                padding='do_not_pad',
                                truncation=True,
                                add_special_tokens=False,
                                max_length=max_text_len)
    payload = prefix_encoding['input_ids']
    if len(payload) > max_text_len - 2:
        payload = payload[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload

    with torch.no_grad():
        result = model({
            'image': img,
            'prefix': torch.tensor(input_ids).unsqueeze(0).cuda(),
        })
    cap = tokenizer.decode(result['predictions'][0].tolist(),
                           skip_special_tokens=True)
    logging.info('output: {}'.format(cap))


# '''
def get_image_transform(param):
    crop_size = param.get('test_crop_size', 224)
    if 'test_respect_ratio_max' in param:
        trans = [
            MinMaxResizeForTest(crop_size, param['test_respect_ratio_max'])
        ]
    else:
        trans = [
            Resize(crop_size, interpolation=Image.BICUBIC),
            CenterCrop(crop_size),
            lambda image: image.convert("RGB"),
        ]
    trans.extend([
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    transforms = Compose(trans)
    return transforms


# '''


def test_git_inference_single_tsv(image_tsv, model_name, question_tsv,
                                  out_tsv):
    param = {}
    if File.isfile(f'output/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(f'output/{model_name}/parameter.yaml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    image_tsv = TSVFile(image_tsv)
    question_tsv = TSVFile(question_tsv) if question_tsv else None

    transforms = get_image_transform(param)

    # model
    model = get_git_model(tokenizer, param)
    pretrained = f'output/{model_name}/snapshot/model.pt'
    checkpoint = torch_load(pretrained)['model']
    load_state_dict(model, checkpoint)
    model.eval()

    torch.cuda.set_device(get_mpi_local_rank())
    model.cuda()

    # prefix
    max_text_len = 40
    rank = get_mpi_rank()
    world_size = get_mpi_size()

    def get_rank_specific_tsv(rank):
        return '{}.{}.{}.tsv'.format(out_tsv, rank, world_size)

    if world_size > 1:
        curr_out_tsv = get_rank_specific_tsv(rank)
    else:
        curr_out_tsv = out_tsv
    total = len(image_tsv)
    curr_size = (total + world_size - 1) // world_size
    curr_start = curr_size * rank
    curr_end = curr_start + curr_size
    curr_end = min(curr_end, total)

    if question_tsv:

        def gen_rows():
            for i in tqdm(range(curr_start, curr_end)):
                image_key, image_col = image_tsv[i]
                q_key, q_info = question_tsv[i]
                assert image_key == q_key
                q_info = json.loads(q_info)
                img = pilimg_from_base64(image_col)
                img = transforms(img)
                img = img.cuda().unsqueeze(0)
                for q in q_info:
                    prefix = q['question']
                    prefix_encoding = tokenizer(prefix,
                                                padding='do_not_pad',
                                                truncation=True,
                                                add_special_tokens=False,
                                                max_length=max_text_len)
                    payload = prefix_encoding['input_ids']
                    if len(payload) > max_text_len - 2:
                        payload = payload[-(max_text_len - 2):]
                    input_ids = [tokenizer.cls_token_id] + payload
                    with torch.no_grad():
                        result = model({
                            'image':
                            img,
                            'prefix':
                            torch.tensor(input_ids).unsqueeze(0).cuda(),
                        })
                    answer = tokenizer.decode(
                        result['predictions'][0].tolist(),
                        skip_special_tokens=True)
                    result = {
                        'answer': answer,
                        'question_id': q['question_id']
                    }
                    yield json_dump(result),
    else:

        def gen_rows():
            for i in tqdm(range(curr_start, curr_end)):
                key, col = image_tsv[i]
                img = pilimg_from_base64(col)
                img = transforms(img)
                img = img.cuda().unsqueeze(0)
                with torch.no_grad():
                    result = model({
                        'image': img,
                    })
                cap = tokenizer.decode(result['predictions'][0].tolist(),
                                       skip_special_tokens=True)
                yield key, json_dump([{'caption': cap}])

    tsv_writer(gen_rows(), curr_out_tsv)
    if world_size > 1 and rank == 0:
        all_sub_tsv = [get_rank_specific_tsv(i) for i in range(world_size)]
        while True:
            not_ready = [t for t in all_sub_tsv if not File.isfile(t)]
            if len(not_ready) == 0:
                break
            else:
                import time
                logging.info('waiting {}'.format(','.join(not_ready)))
                time.sleep(5)
        from .tsv_io import concat_tsv_files
        concat_tsv_files(all_sub_tsv, out_tsv)


def convert_tsv_to_vqa_json(predict_file, out_json):
    result = [json.loads(s) for s, in tsv_reader(predict_file)]
    write_to_file(json_dump(result), out_json)


def convert_tsv_to_coco_format(res_tsv,
                               outfile,
                               sep='\t',
                               key_col=0,
                               cap_col=1):
    results = []
    with open(res_tsv) as fp:
        for line in fp:
            parts = line.strip().split(sep)
            key = parts[key_col]
            if cap_col < len(parts):
                caps = json.loads(parts[cap_col])
                if len(caps) == 0:
                    caps = [{'caption': ''}]
                assert len(
                    caps) == 1, 'cannot evaluate multiple captions per image'
                cap = caps[0]['caption']
            else:
                # empty caption generated
                cap = ""
            results.append({'image_id': key, 'caption': cap})
    with open(outfile, 'w') as fp:
        json.dump(results, fp)


def iter_caption_to_json(iter_caption, json_file):
    # save gt caption to json format so thet we can call the api
    key_captions = [(key, json.loads(p)) for key, p in iter_caption]

    info = {
        'info': 'dummy',
        'licenses': 'dummy',
        'type': 'captions',
    }
    info['images'] = [{'file_name': k, 'id': k} for k, _ in key_captions]
    n = 0
    annotations = []
    for k, cs in key_captions:
        for c in cs:
            annotations.append({
                'image_id': k,
                'caption': c['caption'],
                'id': n
            })
            n += 1
    info['annotations'] = annotations
    write_to_file(json.dumps(info), json_file)


def evaluate_on_coco_caption(
    res_file,
    label_file,
    outfile=None,
):
    if not outfile:
        outfile = op.splitext(res_file)[0] + '.eval.json'

    if res_file.endswith('.tsv'):
        res_file_coco = op.splitext(res_file)[0] + '_coco_format.json'
        convert_tsv_to_coco_format(res_file, res_file_coco)
    else:
        res_file_coco = res_file

    if label_file.endswith('.tsv'):
        json_caption = '/tmp/{}'.format(label_file)
        iter_caption_to_json(TSVFile(label_file), json_caption)
        label_file = json_caption

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file_coco)
    cocoEval = COCOEvalCap(coco, cocoRes)

    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result


if __name__ == '__main__':
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
