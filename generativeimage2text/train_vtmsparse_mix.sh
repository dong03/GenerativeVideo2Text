TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=$2 --master_port=$3  --use_env train_like_mplug.py \
    --config ../config/train_vtmsparse_mix.yaml \
    --output_dir ../ckpt/caption_git_large_dcb_bv/vtmsparsefn_lr_mix$4$(date +'%d-%H-%M')/ \
    --do_two_optim \
    --min_length 15 \
    --beam_size 10 \
    --max_length 32 \
    --max_input_length 48
