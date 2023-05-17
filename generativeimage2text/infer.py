from torch_common import load_state_dict
from model import get_git_model
from dataset import create_dataset, create_sampler, create_loader, cap_collate_fn
import utils
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
import torch.backends.cudnn as cudnn
import torch
from decord import VideoReader
from pathlib import Path
import random
import numpy as np
import ruamel.yaml as yaml
import os
import argparse
import matplotlib
matplotlib.use('Agg')


def load_images(test_dir, test_transform, config):
    image_files = os.listdir(test_dir)
    image_files = sorted(
        image_files,
        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    image_files = [
        os.path.join(test_dir, each) for each in image_files
    ]
    if len(image_files) != config['num_frm_test']:
        sample_ix = np.linspace(0,
                                len(image_files) - 1,
                                num=config['num_frm_test'],
                                endpoint=True,
                                retstep=False,
                                dtype=int)
        image_files = np.array(image_files)[sample_ix].tolist()
    images = [Image.open(i).convert('RGB') for i in image_files]

    images = [test_transform(i) for i in images]
    images = torch.stack(images)
    return images


def load_video(video_path, height=None, width=None, start_time=None, end_time=None, fps=-1, num_frm=16, test_transform=None):

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

    frame_indices = np.arange(
        start_idx, end_idx, vlen / num_frm, dtype=int)

    raw_sample_frms = vr.get_batch(frame_indices).numpy()
    # raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2).numpy()

    raw_sample_frms = [Image.fromarray(each) for each in raw_sample_frms]
    images = [test_transform(i) for i in raw_sample_frms]
    images = torch.stack(images)
    return images


@torch.no_grad()
def infer_single(model, tokenizer, device, config):
    model.eval()
    test_file = config['test_file']
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    test_transform = transforms.Compose([
        transforms.Resize(
            (config['image_res'], config['image_res']), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    if os.path.isdir(test_file):
        images = load_images(test_file, test_transform, config)
    else:
        images = load_video(
            test_file, num_frm=config['num_frm_test'], test_transform=test_transform)

    images = images.to(device, non_blocking=True)
    images = images.unsqueeze(0)
    caption = tokenizer([''], padding='longest', truncation=True,
                        max_length=config['max_input_length'], return_tensors="pt").to(device)
    if 'prefix' in config:
        prefix = tokenizer(config['prefix'], padding='longest', truncation=True,
                           max_length=config['max_input_length'], return_tensors="pt").to(device)
        prefix = prefix['input_ids'][:, :-1]

    input_data = {
        'image': images,
        'need_predict': caption['attention_mask'],
        'caption_tokens': caption['input_ids'],
    }
    if 'prefix' in config:
        input_data['prefix'] = prefix

    result = model(input_data)
    # for i in range(result['predictions'].shape[0]):
    cap = tokenizer.decode(
        result['predictions'][0],
        skip_special_tokens=True)
    cap = cap.replace(
        "[CLS]", "").replace("[PAD]", "").strip()
    print(test_file)
    print('===============')
    print(cap)
    return cap


@torch.no_grad()
def infer_batch(model, data_loader, tokenizer, device, config):
    # test;
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate caption test result:'
    print_freq = 50

    ral_val = []

    for n, (image, image_names) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # if n == 5:
        #     break
        image = image.to(device, non_blocking=True)
        caption = ['' for _ in range(image.shape[0])]
        caption = tokenizer('', padding='longest', truncation=True,
                            max_length=config['max_input_length'], return_tensors="pt").to(device)
        if 'prefix' in config:
            prefix = tokenizer(config['prefix'], padding='longest', truncation=True,
                               max_length=config['max_input_length'], return_tensors="pt").to(device)
            prefix = prefix['input_ids'][:, :-1]

        for i in range(len(image_names)):
            input_data = {
                'image': image[i:i+1],
                'need_predict': caption['attention_mask'][i:i+1],
                'caption_tokens': caption['input_ids'][i:i+1],
            }
            if 'prefix' in config:
                input_data['prefix'] = prefix

            result = model(input_data)

            cap = tokenizer.decode(
                result['predictions'][0],
                skip_special_tokens=True)
            cap = cap.replace(
                "[CLS]", "").replace("[PAD]", "").strip()
            ral_val.append({
                "question_id": image_names[i],
                "pred_caption": cap,
            })
    return ral_val


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    print("Total: %.2fM, Trainable: %.2fM" %
          (total_num / 1.0e6, trainable_num / 1.0e6))


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(
        "uer/gpt2-chinese-cluecorpussmall")
    model = get_git_model(tokenizer, {}, config)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # model.load_state_dict(checkpoint)
    load_state_dict(model, checkpoint)

    model.eval()
    model = model.to(device)
    #get_parameter_number(model)
    if '.txt' in config['test_file']:
        dataset = create_dataset(config=config)

        sampler = None
        test_loader = create_loader(dataset, sampler,
                                    batch_size=config['batch_size_test'],
                                    n_worker=32,
                                    collate_fn=cap_collate_fn)

        result = infer_batch(
            model, test_loader, tokenizer, device, config)
        # import pdb; pdb.set_trace()
        save_file_name = os.path.split(
            args.checkpoint)[-1].split('.')[0] + '_' + os.path.split(config['test_file'][0])[-1].split('.')[0]
        if 'prefix' in config:
            save_file_name += config['prefix'].replace(' ', '-')

        with open(os.path.join(args.output_dir, save_file_name + '.txt'), 'w') as f:
            for res in result:
                f.write(f"{res['question_id']}\t{res['pred_caption']}\n")
    else:
        infer_single(model, tokenizer, device, config)
    # result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch10')
    # if utils.is_main_process():
    #     result = cal_metric(result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--max_input_length', default=25, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--add_object', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--accum_steps', default=4, type=int)
    parser.add_argument('--to_be_infered', default='', type=str)
    parser.add_argument('--git', action='store_true')
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--test_root', default="../demo/frames",type=str)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config['test_file'] = args.to_be_infered
    config['max_input_length'] = args.max_input_length
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["add_object"] = args.add_object
    config["beam_size"] = args.beam_size
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder
    config['test_root'] = args.test_root
    config['use_video'] = False
    if args.git:
        config['gvt'] = False
    if args.use_video:
        config['use_video'] = True

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
