TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=$2 --master_port=3227  --use_env train_like_mplug.py \
    --config /home/dcb/code/bv/git_aimc/config/train_bv_title.yaml \
    --output_dir /home/dcb/code/bv/git_aimc/ckpt/caption_git_large_dcb_bv/freezeimage_random_captitle$(date +'%d-%H-%M')/ \
    --do_two_optim \
    --min_length 15 \
    --beam_size 10 \
    --max_length 32 \
    --max_input_length 48
