AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference_dcb -p \
"{'type': 'each_dcb_inference_single_image', \
      'model_name': 'GIT_BASE_VATEX', \
      'prefix': '', \
      'gpu_id': '0', \
      'ckpt': './output/git_randombv.pth'
}"


# AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference_dcb -p \
# "{'type': 'each_dcb_inference_single_image', \
#       'model_name': 'GIT_BASE_VATEX', \
#       'prefix': '', \
#       'gpu_id': '0', \
#       'prefix': 'what is it?'
# }"

# AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference_dcb -p \
# "{'type': 'each_dcb_inference_single_image', \
#       'model_name': 'GIT_BASE_VQAv2', \
#       'prefix': '', \
#       'gpu_id': '0',
# }"

# AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference_dcb -p \
# "{'type': 'each_dcb_inference_single_image', \
#       'model_name': 'GIT_BASE_VQAv2', \
#       'prefix': '', \
#       'gpu_id': '0', \
#       'prefix': 'what is it?'
# }"

# model_name: 'GIT_LARGE_COCO', 'GIT_LARGE_TEXTCAPS', 'GIT_LARGE_VATEX'


