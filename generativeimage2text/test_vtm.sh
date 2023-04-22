TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1 python infer_like_mplug.py \
    --config ../config/train_vtmsparse_bv.yaml \
    --output_dir ../ckpt/results/ \
    --checkpoint ../ChinaOpenCkpt/GVT_ft_ChinaOpen.pth \
    --min_length 15 \
    --beam_size 10 \
    --max_length 32 \
    --max_input_length 48 \
    --vtm_file ../data/cmo_total_cntags_0422.txt