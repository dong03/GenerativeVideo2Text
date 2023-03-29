AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p \
"{'type': 'each_dcb_inference_single_image', \
      'model_name': 'GIT_LARGE_VATEX', \
      'prefix': '', \
      'gpu_id': '0', \
}"
AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p \
"{'type': 'each_dcb_inference_single_image', \
      'model_name': 'GIT_LARGE', \
      'prefix': '', \
      'gpu_id': '0', \
}"
# model_name: 'GIT_LARGE_COCO', 'GIT_LARGE_TEXTCAPS', 'GIT_LARGE_VATEX'


