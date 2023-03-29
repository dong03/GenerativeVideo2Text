import streamlit as st
import tempfile
import cv2
import json
import os
from torch.cuda.amp import autocast
import numpy as np
import os.path as op
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .common import qd_tqdm as tqdm
from .common import json_dump, Config
from .common import pilimg_from_base64
from .common import get_mpi_rank, get_mpi_size, get_mpi_local_rank

from .tsv_io import TSVFile, tsv_writer, tsv_reader
from .common import write_to_file
import torch
import PIL
from pprint import pformat
import logging
from transformers import BertTokenizer, ChineseCLIPProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from azfuse import File
from .train import get_transform_image
from .common import init_logging
from .common import parse_general_args
from .tsv_io import load_from_yaml_file
from .torch_common import torch_load
from .torch_common import load_state_dict
from .process_image import load_image_by_pil
from .model import get_git_model

def get_model(model_name = 'GIT_LARGE_VATEX',prefix = ''):
    device = torch.device(f"cuda:0")
    param = {}
    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(
            f'aux_data/models/{model_name}/parameter.yaml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    transforms = get_image_transform(param)
    
    model = get_git_model(tokenizer, param)
    pretrained = f'output/{model_name}/snapshot/model.pt'
    # import pdb; pdb.set_trace()
    checkpoint = torch.load(pretrained, map_location='cpu')['model']
    load_state_dict(model, checkpoint)
    model.to(device)
    model.eval()
    
    max_text_len = 40
    prefix_encoding = tokenizer(prefix,
                                padding='do_not_pad',
                                truncation=True,
                                add_special_tokens=False,
                                max_length=max_text_len)
    payload = prefix_encoding['input_ids']
    if len(payload) > max_text_len - 2:
        payload = payload[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload

    prefix = torch.tensor(input_ids).unsqueeze(0).to(device)
    return model, prefix


if __name__ == '__main__':
    st.title("Video Captioning Model")

    st.write("Upload your video and Generate Caption automatically")

    model, prefix = get_model()

    file = st.file_uploader("请上传你要描述的mp4视频文件", type=["mp4"])


    if st.button("Generate Caption"):
        if file is None:
            st.write("Insert image to generate Caption")
            
        else:
            st.video(file)
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            vf = cv2.VideoCapture(tfile.name)
            suc, img = vf.read()
            st.write(img)
            # vf = cv.VideoCapture(tfile.name)
            # prediction = model.predict(image)
            # prediction = prediction.strip(".").strip()
            # st.write(prediction)