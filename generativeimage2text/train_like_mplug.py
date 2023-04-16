from apex import amp
from torch_common import torch_load, load_state_dict
from model import get_git_model
from optim import create_optimizer, create_two_optimizer
from scheduler import create_scheduler
from dataset import create_dataset, create_sampler, create_loader, cap_collate_fn
from dataset.utils import save_result
import utils
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, GPT2Model, AutoTokenizer
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


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=False, accum_steps=1):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(
            window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(
            window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(
            window_size=50, fmt='{value:.6f}'))
    if config['vtm']:
        metric_logger.add_meter('loss_cap', utils.SmoothedValue(
            window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_vtm', utils.SmoothedValue(
            window_size=1, fmt='{value:.4f}'))
    else:
        metric_logger.add_meter('loss', utils.SmoothedValue(
            window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    print("====================================")

    for i, (image, image_name, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # metric_logger.update(loss=1.0)
        # continue
        image = image.to(device, non_blocking=True)
        question_input = None

        caption = tokenizer(caption, padding='longest', truncation=True,
                            max_length=args.max_input_length, return_tensors="pt").to(device)

        # question_input = caption.input_ids[0,0].repeat(caption.input_ids.size(0), 1)

        input_data = {
            'image': image,
            'need_predict': caption['attention_mask'],
            'caption_tokens': caption['input_ids'],
        }
        loss_dict = model(input_data)
        loss = sum([v for v in loss_dict.values()])
        # import pdb; pdb.set_trace()
        if accum_steps > 1:
            loss = loss / accum_steps

        if do_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # logger.info('scaled loss: {}'.format(str(scaled_loss)))
                scaled_loss.backward()
        else:
            loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if config['vtm']:
            metric_logger.update(loss_cap=loss_dict['vl_l_caploss'].item())
            metric_logger.update(loss_vtm=loss_dict['vl_l_vtmloss'].item())
        else:
            metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

        del image, question_input, caption, loss
        # if i:
        #     break

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    # '''


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

        # import
        # for image_id, topk_id, topk_prob, gold_caption_list in zip(image_names, topk_ids, topk_probs, caption['input_ids']):
        #     ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace(
        #         "[CLS]", "").replace("[PAD]", "").strip()
        #     result.append({
        #         "question_id": image_id,
        #         "pred_caption": ans,
        #         "gold_caption": tokenizer.decode(gold_caption_list).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()})
        if n == 5:
            break
    return ral_val


def cal_metric(result_file):
    result_list = json.load(open(result_file, "r"))
    predicts = []
    answers = []
    for each in result_list:
        predicts.append(each["pred_caption"])
        answers.append(each["gold_caption"])
    evaluator = language_evaluation.CocoEvaluator(verbose=False)
    results = evaluator.run_evaluation(predicts, answers)
    results['BLEU'] = (results['Bleu_1'] + results['Bleu_2'] +
                       results['Bleu_3'] + results['Bleu_4']) / 4.0
    del results['Bleu_1']
    del results['Bleu_2']
    del results['Bleu_3']
    del results['Bleu_4']
    print(len(result_list), results)
    return results


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

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    # print("Creating vqa datasets")
    datasets = create_dataset('dcb_frames_caps', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader(datasets, samplers,
                                                          batch_size=[
                                                              config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                                          num_workers=[16, 8, 8], is_trains=[True, False, False],
                                                          collate_fns=[cap_collate_fn, cap_collate_fn, cap_collate_fn])

    # tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
    
    # tokenizer = ChineseCLIPProcessor.from_pretrained(
    #     "OFA-Sys/chinese-clip-vit-base-patch16")
    # tokenizer = tokenizer.tokenizer
    model = get_git_model(tokenizer, {}, config)

    pretrained = f'/home/dcb/code/bv/git_aimc/output/{model_name}/snapshot/model.pt'
    checkpoint = torch_load(pretrained)['model']

    # for key in list(checkpoint.keys()):
    #     if 'textual' in key:
    #         del checkpoint[key]
    load_state_dict(model, checkpoint)
    temp_encoder = GPT2Model.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    # temp_encoder = ChineseCLIPModel.from_pretrained(
    #     "OFA-Sys/chinese-clip-vit-base-patch16").text_model
    model.text_encoder = temp_encoder

    if config['freeze'] == 'image':
        for n, p in model.named_parameters():
            if 'image_encoder' in n or 'text_encoder' in n:
                p.requires_grad = False
    elif config['freeze'] == 'all':
        for n, p in model.named_parameters():
            if 'textual.embedding' not in n and 'textual.output' not in n and 'textual.classifier' not in n:
                p.requires_grad = False
    elif config['freeze'] == 'None':
        pass
    get_parameter_number(model)
    model = model.to(device)
    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.do_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    model_without_ddp = model
    if args.distributed:
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model = apex.parallel.DistributedDataParallel(
            model, delay_allreduce=True)
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()
    # vqa_result = evaluation(model, test_loader, tokenizer, device, config)
    # result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch10')
    # if utils.is_main_process():
    #     result = cal_metric(result_file)
    dist.barrier()
    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)

        if utils.is_main_process():
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            print(log_stats)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            vqa_result = evaluation(
                model, test_loader, tokenizer, device, config)
            result_file = save_result(
                vqa_result, args.output_dir, 'bv_result_epoch%d' % epoch)
            '''
            import pdb
            pdb.set_trace()
            result = cal_metric(result_file)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         **{f'val_{k}': v for k, v in result.items()}
                         }
            print(log_stats)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
            # '''

        # dist.barrier()

    # vqa_result = evaluation(model, test_loader, tokenizer, device, config)
    # result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


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
