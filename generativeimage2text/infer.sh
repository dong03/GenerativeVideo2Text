# single video
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1 python infer.py \
    --config ../config/infer.yaml \
    --output_dir ../ckpt/results/ \
    --checkpoint ../GVT_ft_ChinaOpen.pth \
    --min_length 15 \
    --beam_size 10 \
    --max_length 32 \
    --max_input_length 48 \
    --to_be_infered ../VisualSearch/BV_Video/BV1CN411o7WE.mp4\
    --use_video

# single images dir
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1 python infer.py \
    --config ../config/infer.yaml \
    --output_dir ../ckpt/results/ \
    --checkpoint ../GVT_ft_ChinaOpen.pth \
    --min_length 15 \
    --beam_size 10 \
    --max_length 32 \
    --max_input_length 48 \
    --to_be_infered ../demo/BV1CN411o7WE

# batch
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1 python infer.py \
    --config ../config/infer.yaml \
    --output_dir ../demo/results/ \
    --checkpoint ../GVT_ft_ChinaOpen.pth \
    --min_length 15 \
    --beam_size 10 \
    --max_length 32 \
    --max_input_length 48 \
    --to_be_infered ../demo/demo.txt