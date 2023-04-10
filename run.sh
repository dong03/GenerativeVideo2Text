AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference_dcb -p \
"{'type': 'each_dcb_inference_single_image', \
      'model_name': 'GIT_BASE_VATEX', \
      'prefix': '', \
      'gpu_id': '0', \
      'ckpt': '/home/dcb/code/bv/git_aimc/ckpt/caption_git_large_dcb_bv/freezeimage_captitle03-09-22/checkpoint_09.pth'
}"
# AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference_dcb -p \
# "{'type': 'each_dcb_inference_single_image', \
#       'model_name': 'GIT_BASE_VATEX', \
#       'prefix': '', \
#       'gpu_id': '0', \
# }"
# model_name: 'GIT_LARGE_COCO', 'GIT_LARGE_TEXTCAPS', 'GIT_LARGE_VATEX'


