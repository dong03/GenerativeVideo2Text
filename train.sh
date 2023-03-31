CUDA_VISIBLE_DEVICES=$1 python -m generativeimage2text.train_dcb -p "{'type': 'dcb_train', \
'yaml': '/home/dcb/code/bv/git_aimc/config/train_vatex.yaml', \
'description': 'freezeall', \
            }"