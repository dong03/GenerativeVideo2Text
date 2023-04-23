# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1 python infer_like_mplug.py \
#     --config /home/dcb/code/bv/git_aimc/config/train_dense_bv.yaml \
#     --output_dir /home/dcb/code/bv/git_aimc/ckpt/results/ \
#     --checkpoint /home/dcb/code/bv/git_aimc/output/git_densebv.pth \
#     --min_length 15 \
#     --beam_size 10 \
#     --max_length 32 \
#     --max_input_length 48

# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1 python infer_like_mplug.py \
#     --config ../config/train_vtmsparse_bv.yaml \
#     --output_dir ../ckpt/results/ \
#     --checkpoint ../output/vtmsparseln_15.pth \
#     --min_length 15 \
#     --beam_size 10 \
#     --max_length 32 \
#     --max_input_length 48

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1 python infer_like_mplug.py \
    --config ../config/train_sparse_bv.yaml \
    --output_dir ../ckpt/results/ \
    --checkpoint ../output/gvt_vtmsparse_semi.pth \
    --min_length 15 \
    --beam_size 10 \
    --max_length 32 \
    --max_input_length 48
