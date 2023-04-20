TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1 python infer_like_mplug.py \
    --config ../config/infer_tagging.yaml \
    --output_dir ../ckpt/results/ \
    --checkpoint ../output/git_fi_captitle.pth \
    --min_length 1 \
    --beam_size 10 \
    --max_length 5 \
    --max_input_length 48
