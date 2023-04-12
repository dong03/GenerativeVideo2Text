import os
import shutil


root = 'ckpt/caption_git_large_dcb_bv'

for each in os.listdir(root):
    if len(os.listdir(os.path.join(root, each))) == 2:
        print(each)
        shutil.rmtree(os.path.join(root, each))