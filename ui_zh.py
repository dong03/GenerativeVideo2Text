import streamlit as st
import tempfile
import numpy as np
import torchvision.transforms as transforms

import torch
from transformers import BertTokenizer
from PIL import Image
from azfuse import File
from generativeimage2text.tsv_io import load_from_yaml_file
from generativeimage2text.torch_common import load_state_dict
from generativeimage2text.model import get_git_model
from generativeimage2text.inference import get_image_transform
from decord import VideoReader
import json

from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service
from tqdm import tqdm
from transformers import BertTokenizer, ChineseCLIPProcessor



def _load_video_from_path_decord(video_path,
                                 height=None,
                                 width=None,
                                 start_time=None,
                                 end_time=None,
                                 fps=-1):
    num_frm = 4
    try:
        if not height or not width:
            vr = VideoReader(video_path)
        else:
            vr = VideoReader(video_path, width=width, height=height)

        vlen = len(vr)

        if start_time or end_time:
            assert fps > 0, 'must provide video fps if specifying start and end time.'

            start_idx = min(int(start_time * fps), vlen)
            end_idx = min(int(end_time * fps), vlen)
        else:
            start_idx, end_idx = 0, vlen

        frame_indices = np.arange(start_idx,
                                  end_idx,
                                  vlen / num_frm,
                                  dtype=int)

        raw_sample_frms = vr.get_batch(frame_indices).asnumpy()
    except Exception as e:
        print(f"fail loading {video_path}")
        return np.zeros((num_frm, 256, 256, 3))
    return raw_sample_frms


@st.cache_resource
def get_model(model_name='GIT_BASE_VATEX', prefix=''):
    device = torch.device(f"cpu")
    param = {}
    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(
            f'aux_data/models/{model_name}/parameter.yaml')

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
    #                                          do_lower_case=True)
    tokenizer = ChineseCLIPProcessor.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16")
    tokenizer = tokenizer.tokenizer
    transforms = get_image_transform(param)

    model = get_git_model(tokenizer, param)
    # pretrained = f'output/{model_name}/snapshot/model.pt'
    # import pdb; pdb.set_trace()
    pretrained = '/home/dcb/code/bv/GenerativeImage2Text/output/git_randombv.pth'
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
    return model, prefix, transforms, device, tokenizer


def predict(tfile):
    images = _load_video_from_path_decord(tfile.name)
    images = [transforms(Image.fromarray(i)) for i in images]
    images = [i.unsqueeze(0).to(device) for i in images]

    result = model({
        'image': images,
        'prefix': prefix,
    })
    ress = tokenizer.decode(result['predictions'][0].tolist(),
                            skip_special_tokens=True)
    return ress


model, prefix, transforms, device, tokenizer = get_model()

if __name__ == '__main__':

    st.title("Video Captioning Demo")

    st.write("上传视频，生成自动描述")

    file = st.file_uploader("请上传你要描述的mp4视频文件", type=["mp4"])
    if file:

        # if st.button("生成描述语句"):
        #     if file is None:
        #         st.write("请上传视频")
        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(file.read())
        ress = predict(tfile)
        # st.write(ress)
        # st.set_page_config(layout="wide")
        st.markdown("""
        <style>
        .big-font {
            font-size:32px;
        }
        </style>
        """,
                    unsafe_allow_html=True)

        st.markdown(f'<p class="big-font">{ress}</p>', unsafe_allow_html=True)
        st.video(file)
