# GVT
- 模型结构改动：```generativeimage2text/layers/gvt_decoder.py```
- 数据集改动：```generativeimage2text/dataset/video_dataset_dcb.py```
- 训练代码：```generativeimage2text/train_like_mplug.py```
- 推理代码：```generativeimage2text/infer_like_mplug.py```
- 训练脚本：
  ```bash
  cd generativeimage2text
  bash vtmsparse.sh GPU_ID GPU_Number Port
  # e.g. bash vtmsparsh.sh 0,1,2,3 4 3520
  ```
- 推理（Captioning）
  ```bash
  cd generativeimage2text
  bash test.sh GPU_ID
  # e.g. bash test.sh 0
  ```
- 推理（Video-text matching）
  ```bash 
  cd generativeimage2text
  bash test_vtm.sh GPU_ID
  # e.g. bash test_vtm.sh 0
  ```
  
- 配置文件：```config/train_vtmsparse_mix.yaml```
  ```yaml
  train_file: list，每个文件为训练信息索引  videoid\tcaption\n
  train_root: list，对应相同顺序的文件的帧的储存位置 train_root/videoid为存放帧的文件夹
  num_frm_test: 每个视频加载多少帧
  vtm: True/False，是否具有video text matching
  dense: True/False，是否密集采帧
  sparse: True/False，是否稀疏采帧
  ```
- ckpt存储路径：```output/```
- 推理结果路径：```ckpt/results```
- 训练细节：
  - 冻住image-encoder
  - 采用sparse visual token：对中间帧保留完整的197个token，其余帧仅使用[CLS] token，增加采样帧数（6帧->16帧）
  - 取last text token + fc 做video-text matching的二分类，通过重新排列组合输入视频-描述语句构建负样本
  - 总计训练15个epoch，学习率为5e-5，采用warmup、衰减等策略

- 结果：

| Captioning     |  config                                     |  模型ckpt                                        | 结果                                                          |
|----------------|---------------------------------------------|------------------------------------------------|-------------------------------------------------------------|
|  GIT(VATEX CN) | [train_vatex.yaml](config/train_vatex.yaml) | [GIT_ft_Vatex.pth](ChinaOpenCkpt/GIT_ft_Vatex.pth) | [GIT_ft_Vatex.txt](ChinaOpenResults/GIT_ft_Vatex.txt)       |
| GIT(RandomBV)  | [train_bv.yaml](config/train_bv.yaml)       | [GIT_ft_RandomBV.pth](ChinaOpenCkpt/GIT_ft_RandomBV.pth)                         | [GIT_ft_RandomBV.txt](ChinaOpenResults/GIT_ft_ChinaOpen.txt) |
| GIT(ChinaOpen)  | [train_bv.yaml](config/train_bv.yaml)       | [GIT_ft_ChinaOpen.pth](ChinaOpenCkpt/GIT_ft_ChinaOpen.pth)                         | [GIT_ft_ChinaOpen.txt](ChinaOpenResults/GIT_ft_ChinaOpen.txt) |
| GVT(ChinaOpen,w/o VTM)  | [train_sparse_bv.yaml](config/train_sparse_bv.yaml)       | [GVT_ft_ChinaOpen_woVTM.pth](ChinaOpenCkpt/GVT_ft_ChinaOpen_woVTM.pth)                         | [GVT_ft_ChinaOpen_woVTM.txt](ChinaOpenResults/GVT_ft_ChinaOpen_woVTM.txt) |
| GVT(ChinaOpen)  | [train_vtmsparse_bv.yaml](config/train_vtmsparse_bv.yaml)       | [GVT_ft_ChinaOpen.pth](ChinaOpenCkpt/GVT_ft_ChinaOpen.pth)                         | [GVT_ft_ChinaOpen.txt](ChinaOpenResults/GVT_ft_ChinaOpen.txt) |


| Matching(v2t)     |  config                                     |  模型ckpt                                        | 结果                                                          |
|----------------|---------------------------------------------|------------------------------------------------|-------------------------------------------------------------|
|  GVT(ChinaOpen, content-based) | [train_vtmsparse_bv.yaml](config/train_vtmsparse_bv.yaml) | [GVT_ft_ChinaOpen.pth](ChinaOpenCkpt/GVT_ft_ChinaOpen.pth) | [GVT_ChinaOpen_caption.npy](ChinaOpenResults/GVT_ChinaOpen_caption.npy)       |
|  GVT(ChinaOpen, content-beyond) | [train_vtmsparse_bv.yaml](config/train_vtmsparse_bv.yaml) | [GVT_ft_ChinaOpen.pth](ChinaOpenCkpt/GVT_ft_ChinaOpen.pth) | [GVT_ChinaOpen_title.npy](ChinaOpenResults/GVT_ChinaOpen_title.npy)       |
|  GVT(ChinaOpen, content-beyond) | [train_vtmsparse_bv.yaml](config/train_vtmsparse_bv.yaml) | [GVT_ft_ChinaOpen.pth](ChinaOpenCkpt/GVT_ft_ChinaOpen.pth) | [GVT_ChinaOpen_tagging.npy](ChinaOpenResults/GVT_ChinaOpen_tagging.npy)       |

# Introduction
This repo presents some example codes to reproduce some results in
[GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100).

# Installation
- Install [azfuse](https://github.com/microsoft/azfuse). The tool is used to
  automatically download the data. The configuration of
  AzFuse has already been in this repo.

- Download the source code by
  ```shell
  git clone https://github.com/microsoft/GenerativeImage2Text.git
  cd GenerativeImage2Text
  ```

- Install the package
  ```shell
  pip install -r requirements.txt
  python setup.py build develop
  ```

# Inference
- Inference on a single image or multiple frames:
  ```shell
  # single image, captioning
  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
        'image_path': 'aux_data/images/1.jpg', \
        'model_name': 'GIT_BASE', \
        'prefix': '', \
  }"
  # single image, question answering
  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
        'image_path': 'aux_data/images/1.jpg', \
        'model_name': 'GIT_BASE_VQAv2', \
        'prefix': 'what is it?', \
  }"
  # multiple images, captioning
  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
        'image_path': ['aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg'], \
        'model_name': 'GIT_BASE_VATEX', \
        'prefix': '', \
  }"
  # multiple images, question answering
  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
        'image_path': ['aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg'], \
        'model_name': 'GIT_BASE_MSRVTT_QA', \
        'prefix': 'what is it?', \
  }"
  ```
  - If `prefix` is empty, it is effectively the captioning task.
  - If `prefix` is a question, it is effectively the visual question answering task.
  - Use a list for `image_path` if it is for video. The example here is 6 identical images, only
    for a demo purpose. It should be different image frames from a video.
  - `model_name` here can be the following. Performance details can be found in the reference paper.

    | model_name          | Information                                   | Performance             |
    |---------------------|---------------------------------------------- | ----------------------- |
    | GIT_BASE            | pretrained on 4M images                       |                         |
    | GIT_BASE_COCO       | fine-tuned on COCO                            | CIDEr: 131.4            |
    | GIT_BASE_TEXTCAPS   | fine-tuned on TextCaps for captioning         | CIDEr: 64.9             |
    | GIT_BASE_VQAv2      | fine-tuned on VQAv2                           | test-dev: 72.72         |
    | GIT_BASE_TEXTVQA    | fine-tuned on TextVQA                         | val/acc: 18.81          |
    | GIT_BASE_VATEX      | fine-tuned on VATEX for captioning            | public/test/CIDEr: 60.0 |
    | GIT_BASE_MSRVTT_QA  | fine-tuned on MSRVTT for question answering   | acc: 41.0               |
    | GIT_LARGE           | pretrained on 14M images                      |                         |
    | GIT_LARGE_COCO      | fine-tuned on COCO                            | CIDEr: 138.5            |
    | GIT_LARGE_TEXTCAPS  | fine-tuned on TextCaps for captioning         | CIDEr: 106.3            |
    | GIT_LARGE_VQAv2     | fine-tuned on VQAv2                           | test-dev: 75.51         |
    | GIT_LARGE_TEXTVQA   | fine-tuned on TextVQA                         | val/acc: 37.47          |
    | GIT_LARGE_VATEX     | fine-tuned on VATEX for captioning            | public/test/CIDEr: 72.5 |
    | GIT_LARGE_MSRVTT_QA | fine-tuned on MSRVTT for question answering   | acc: 42.7               |

- Inference on a [TSV](https://en.wikipedia.org/wiki/Tab-separated_values) file, which is a collection of multiple images.
  - Data format (for information only)
    - image TSV: Each row has two columns. The first is the image key; the
      second is base64-encoded jpg or png bit string.
    - caption or question tsv: Each row has two columns. The first is the image
      key; the second is a list of dictionaries in the json format. For caption TSV,
      the dictionary should contain at least the field of `'caption'`. For the
      question answering TSV, it should contain at least `question_id` and
      `question`.
  - inference on [COCO](https://cocodataset.org) Karpathy test.
      <!---
    1. Prepare the coco test TSV
       ```
       mkdir -p aux_data/raw_data
       wget http://images.cocodataset.org/zips/val2014.zip -O aux_data/raw_data/val2014.zip
       wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip -O aux_data/raw_data/caption_datasets.zip
       cd aux_data/raw_data
       unzip val2014.zip
       unzip caption_datasets.zip
       python -m generativeimage2text.data_prepare -p "{'type': 'prepare_coco_test'}"
       ```
       -->
    1. Inference.
       ```shell
       # base
       AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
             'image_tsv': 'data/coco_caption/test.img.tsv', \
             'model_name': 'GIT_BASE_COCO', \
             'question_tsv': null, \
             'out_tsv': 'inference/GIT_BASE_COCO/coco.tsv', \
       }"
       # GIT_LARGE_COCO. If there are 8 GPUs, it can parallel by mpirun -n 8
       AZFUSE_TSV_USE_FUSE=1 mpirun -n 8 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
             'image_tsv': 'data/coco_caption/test.img.tsv', \
             'model_name': 'GIT_LARGE_COCO', \
             'question_tsv': null, \
             'out_tsv': 'inference/GIT_LARGE_COCO/coco.tsv', \
       }"
       ```
    2. Calculate the evaluation metric
       ```shell
       # base
       AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'evaluate_on_coco_caption', \
             'res_file': 'inference/GIT_BASE_COCO/coco.tsv', \
             'label_file': 'data/coco_caption/test.caption.tsv', \
       }"
       ```
       The CIDEr score should be 131.35 for `GIT_BASE_COCO` and  138.45 for `GIT_LARGE_COCO`.
       If you get lower score (e.g. 126 for the base model),
       the reason could be
       the misalignment of the environment, e.g. pytorch version.
    3. (optional) To exactly reproduce the number, please run the following:
       ```bash
       nvidia-docker run --ipc=host amsword/setup:py38pt19u20cu11 \
           bash -c "mkdir -p /tmp/code \
                   && cd /tmp/code \
                   && pip install git+https://github.com/microsoft/azfuse.git \
                   && git clone https://github.com/amsword/generativeimage2text.git \
                   && cd generativeimage2text \
                   && pip install -r requirements.txt \
                   && python setup.py build develop \
                   && AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
                            'image_tsv': 'data/coco_caption/test.img.tsv', \
                            'model_name': 'GIT_BASE_COCO', \
                            'question_tsv': null, \
                            'out_tsv': 'inference/GIT_BASE_COCO/coco.tsv', \
                      }" \
                   &&  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'evaluate_on_coco_caption', \
                       'res_file': 'inference/GIT_BASE_COCO/coco.tsv', \
                       'label_file': 'data/coco_caption/test.caption.tsv', \
                       'outfile': 'inference/GIT_BASE_COCO/coco.score.json', \
                       }" \
                   && cat inference/GIT_BASE_COCO/coco.score.json \
                   "
       ```
  - Inference on [vqa](https://visualqa.org/index.html) test
    1. Inference
       ```shell
       # base model
       AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
             'image_tsv': 'data/TaxVQAv2/test.tsv', \
             'model_name': 'GIT_BASE_VQAv2', \
             'question_tsv': 'data/TaxVQAv2/test.caption.tsv', \
             'out_tsv': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.tsv', \
       }"
       # GIT_LARGE_VQAv2 with 8 GPUs.
       AZFUSE_TSV_USE_FUSE=1 mpirun -n 8 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
             'image_tsv': 'data/TaxVQAv2/test.tsv', \
             'model_name': 'GIT_LARGE_VQAv2', \
             'question_tsv': 'data/TaxVQAv2/test.caption.tsv', \
             'out_tsv': 'inference/GIT_LARGE_VQAv2/snapshot/vqav2.tsv', \
       }"
       ```

    2. Convert the output tsv to the json format for submission to [evalai](https://eval.ai/web/challenges/challenge-page/830/overview)
       ```shell
       # base model
       AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'convert_tsv_to_vqa_json', \
             'predict_file': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.tsv', \
             'out_json': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.json', \
       }"
       # large model
       AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'convert_tsv_to_vqa_json', \
             'predict_file': 'inference/GIT_LARGE_VQAv2/snapshot/vqav2.tsv', \
             'out_json': 'inference/GIT_LARGE_VQAv2/snapshot/vqav2.json', \
       }"
       ```
       Submit the file of `inference/GIT_BASE_VQAv2/snapshot/vqav2.json` to evalai
       and you should get `72.72` on `test-dev`. If it is `GIT_LARGE_VQAv2`, the accuracy is
       `75.51`.

    3. (optional) To exactly reproduce the number, you can use the
       following:
       ```shell
       # base model
       nvidia-docker run --ipc=host amsword/setup:py38pt19u20cu11 \
           bash -c "mkdir /tmp/code \
                   && cd /tmp/code \
                   && pip install git+https://github.com/microsoft/azfuse.git \
                   && git clone https://github.com/amsword/generativeimage2text.git \
                   && cd generativeimage2text \
                   && pip install -r requirements.txt \
                   && python setup.py build develop \
                   && AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
                       'image_tsv': 'data/TaxVQAv2/test.tsv', \
                       'model_name': 'GIT_BASE_VQAv2', \
                       'question_tsv': 'data/TaxVQAv2/test.caption.tsv', \
                       'out_tsv': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.tsv', \
                   }" \
                   &&  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'convert_tsv_to_vqa_json', \
                       'predict_file': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.tsv', \
                       'out_json': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.json', \
                   }" \
       }"
       ```
       Note that, please modify the docker command properly so that the output
       file can be saved permanently to the host machine. It is also recommended
       to run it inside the docker container by
       ```shell
       nvidia-docker run --ipc=host amsword/setup:py38pt19u20cu11 sleep infinity
       docker ps # get the docker container ID
       docker exec -it container_id /bin/bash # attach inside the docker container
       # all other commands to run the inference.
       ```

# Training
The repo shows the key code path of constructing the network
input with transformations and forward/backward. The code can be plugged into
any trainer easily. Here is the example for the base model.
- Pretraining/captioning
  ```
  python -m generativeimage2text.train -p "{'type': 'forward_backward_example', \
                  'image_files': ['aux_data/images/1.jpg', 'aux_data/images/2.jpg'], \
                  'captions': ['a couple of boats in a large body of water.', 'a view of a mountain with a tree'], \
              }"
  ```
- VQA
  ```
  python -m generativeimage2text.train -p "{'type': 'forward_backward_example', \
                  'image_files': ['aux_data/images/1.jpg', 'aux_data/images/2.jpg'], \
                  'prefixs': ['what is this?', 'how many trees?'], \
                  'captions': ['several boats in a large body of water', '1'], \
              }"
  ```


# ImageNet
## Class ID to unique readable names
- Save the file of `LOC_synset_mapping.txt` from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=LOC_synset_mapping.txt).
  under `aux_data/imagenet/`

- Convert the wordnet ID to readable names as follows
  ```python
  python -m generativeimage2text.data_prepare -p "{'type': 'generate_imagenet_unique_names'}"
  ```
  The input file is hard coded as `./aux_data/imagenet/LOC_synset_mapping.txt` and the
  output file is `./aux_data/imagenet/imagenet_unique_readable_names.txt`

# Citation
Please consider to cite the following reference if it helps.
```text
@article{wang2022git,
  title={GIT: A Generative Image-to-text Transformer for Vision and Language},
  author={Wang, Jianfeng and Yang, Zhengyuan and Hu, Xiaowei and Li, Linjie and Lin, Kevin and Gan, Zhe and Liu, Zicheng and Liu, Ce and Wang, Lijuan},
  journal={arXiv preprint arXiv:2205.14100},
  year={2022}
}
```

# Acknowledgement
Part of the code is based on
[transformers](https://github.com/huggingface/transformers),
[clip](https://github.com/openai/CLIP),
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark),
[oscar](https://github.com/microsoft/Oscar),
[virtex](https://github.com/kdexd/virtex).


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
