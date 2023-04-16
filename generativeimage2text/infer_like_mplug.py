from apex import amp
from torch_common import torch_load, load_state_dict
from model import get_git_model
from optim import create_optimizer, create_two_optimizer
from scheduler import create_scheduler
from dataset import create_dataset, create_sampler, create_loader, cap_collate_fn
from dataset.utils import save_result
import utils
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch
from pathlib import Path
import json
import datetime
import time
import random
import numpy as np
import language_evaluation
import ruamel.yaml as yaml
import os
import argparse
import matplotlib
from apex import amp
import apex
matplotlib.use('Agg')


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate caption test result:'
    print_freq = 5

    ral_val = []

    answer_input = None
    for n, (image, image_names, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)

        caption = tokenizer(caption, padding='longest', truncation=True,
                            max_length=args.max_input_length, return_tensors="pt").to(device)
        from tqdm import tqdm
        for i in tqdm(range(len(image_names))):
            input_data = {
                'image': image[i:i+1],
                'need_predict': caption['attention_mask'][i:i+1],
                'caption_tokens': caption['input_ids'][i:i+1],
            }
            result = model(input_data)
            cls_prob = result.get('cls_prob', torch.tensor([[0.0, 0.0]]))
        # for i in range(result['predictions'].shape[0]):
            cap = tokenizer.decode(
                result['predictions'][0],
                skip_special_tokens=True)
            cap = cap.replace(
                "[CLS]", "").replace("[PAD]", "").strip()
            ral_val.append({
                "question_id": image_names[i],
                "pred_caption": cap,
                "gold_caption": tokenizer.decode(caption['input_ids'][i], skip_special_tokens=True).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip(),
                "vtm_score": cls_prob[0, 1].item()})
            import pdb
            pdb.set_trace()
        # import
        # for image_id, topk_id, topk_prob, gold_caption_list in zip(image_names, topk_ids, topk_probs, caption['input_ids']):
        #     ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace(
        #         "[CLS]", "").replace("[PAD]", "").strip()
        #     result.append({
        #         "question_id": image_id,
        #         "pred_caption": ans,
        #         "gold_caption": tokenizer.decode(gold_caption_list).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()})

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

    #### Dataset ####
    # print("Creating vqa datasets")
    datasets = create_dataset('dcb_frames_caps', config)

    samplers = [None, None, None]
    _, _, test_loader = create_loader(datasets, samplers,
                                      batch_size=[
                                          config['batch_size_train'], config['batch_size_test'], config['batch_size_test']*4],
                                      num_workers=[32, 8, 8], is_trains=[True, False, False],
                                      collate_fns=[cap_collate_fn, cap_collate_fn, cap_collate_fn])

    # tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    tokenizer = ChineseCLIPProcessor.from_pretrained(
        "OFA-Sys/chinese-clip-vit-base-patch16")
    tokenizer = tokenizer.tokenizer
    model = get_git_model(tokenizer, {}, config)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')['model']
    # model.load_state_dict(checkpoint)
    load_state_dict(model, checkpoint)
    # temp_encoder = ChineseCLIPModel.from_pretrained(
    #     "OFA-Sys/chinese-clip-vit-base-patch16").text_model
    # model.text_encoder = temp_encoder

    model.eval()
    model = model.to(device)
    get_parameter_number(model)
    result = evaluation(
        model, test_loader, tokenizer, device, config)

    save_file_name = os.path.split(
        args.checkpoint)[-1].split('.')[0] + '_' + os.path.split(config['test_file'])[-1]
    with open(os.path.join(args.output_dir, save_file_name), 'w') as f:
        for res in result:
            f.write(f"{res['question_id']}\t{res['pred_caption']}\n")

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
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["add_object"] = args.add_object
    config["beam_size"] = args.beam_size
    #config['optimizer']['lr'] = args.lr
    #config['schedular']['lr'] = args.lr
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
