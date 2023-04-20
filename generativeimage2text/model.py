import torch
try:
    from .torch_common import resize_2d_pos_embed
    from .layers.CLIP import clip
    from .layers.decoder import CaptioningModel
    from .layers.decoder import (TransformerDecoderTextualHead,
                                 AutoRegressiveBeamSearch, GeneratorWithBeamSearch)
    from .layers.dcb_decoder import CaptioningVTMModel, CaptioningDenseModel, CaptioningVTMDenseModel, CaptioningSparseModel, CaptioningVTMSparseModel, TransformerDecoderClfTextualHead

except:
    from torch_common import resize_2d_pos_embed
    from layers.CLIP import clip
    from layers.decoder import CaptioningModel
    from layers.decoder import (TransformerDecoderTextualHead,
                                AutoRegressiveBeamSearch, GeneratorWithBeamSearch)
    from layers.dcb_decoder import CaptioningVTMModel, CaptioningDenseModel, CaptioningVTMDenseModel, CaptioningSparseModel, CaptioningVTMSparseModel, TransformerDecoderClfTextualHead

from transformers import ChineseCLIPModel, GPT2Model


def get_git_model(tokenizer, param, dcb_param=None):
    image_encoder = get_image_encoder(
        param.get('image_encoder_type', 'CLIPViT_B_16'),
        input_resolution=param.get('test_crop_size', 224),
    )
    if dcb_param is None:
        TEXT_DECODER = TransformerDecoderTextualHead
        CAP_MODEL = CaptioningModel
        TEXT_ENCODER = None
    else:
        if dcb_param['vtm'] and dcb_param['dense']:
            TEXT_DECODER = TransformerDecoderClfTextualHead
            CAP_MODEL = CaptioningVTMDenseModel
            TEXT_ENCODER = None
            # TEXT_ENCODER = GPT2Model.from_pretrained(
            #     "uer/gpt2-chinese-cluecorpussmall")
            # TEXT_ENCODER = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").text_model
        elif dcb_param['vtm'] and dcb_param['sparse']:
            TEXT_DECODER = TransformerDecoderClfTextualHead
            CAP_MODEL = CaptioningVTMSparseModel
            TEXT_ENCODER = None
            # TEXT_ENCODER = GPT2Model.from_pretrained(
            #     "uer/gpt2-chinese-cluecorpussmall")
        elif not dcb_param['vtm'] and not dcb_param['dense'] and not dcb_param['sparse']:
            TEXT_DECODER = TransformerDecoderTextualHead
            CAP_MODEL = CaptioningModel
            TEXT_ENCODER = None
        # not sure, may be need to be delated
        elif not dcb_param['vtm']:
            TEXT_DECODER = TransformerDecoderTextualHead
            CAP_MODEL = CaptioningDenseModel
            TEXT_ENCODER = None
        elif not dcb_param['dense']:
            TEXT_DECODER = TransformerDecoderClfTextualHead
            CAP_MODEL = CaptioningVTMModel
            TEXT_ENCODER = GPT2Model.from_pretrained(
                "uer/gpt2-chinese-cluecorpussmall")
            # ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").text_model
        else:
            raise NotImplementedError
    text_decoder = TEXT_DECODER(
        visual_feature_size=param.get('visual_feature_size', 768),
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_layers=6,
        attention_heads=12,
        feedforward_size=768 * 4,
        max_caption_length=1024,
        mask_future_positions=True,
        padding_idx=0,
        decoder_type='bert_en',
        visual_projection_type='linearLn',
    )
    # decoder = AutoRegressiveBeamSearch(
    # eos_index=tokenizer.sep_token_id,
    # max_steps=40,
    # beam_size=1,
    # per_node_beam_size=1,
    # fix_missing_prefix=True,
    # )
    decoder = GeneratorWithBeamSearch(
        eos_index=tokenizer.sep_token_id,
        # max_steps=40,
        max_steps=1024,
        beam_size=4,
        length_penalty=0.6,
    )

    model = CAP_MODEL(
        image_encoder,
        text_decoder,
        decoder=decoder,
        sos_index=tokenizer.cls_token_id,
        eos_index=tokenizer.sep_token_id,
        tokenizer=tokenizer,
        use_history_for_infer=True,
        loss_type='smooth',
        num_image_with_embedding=param.get('num_image_with_embedding'),
        text_encoder=TEXT_ENCODER
    )
    return model


def get_image_encoder(encoder_type, input_resolution=224):
    name_map = {
        'CLIPViT_B_16': 'ViT-B/16',
        'CLIPViT_L_14': 'ViT-L/14',
    }
    name_in_clip = name_map[encoder_type]
    model, _ = clip.load(name_in_clip, device='cpu', jit=False)
    model = model.train()
    ret = model.visual
    ret.to(torch.float32)
    ret.output_grid = True
    ret.grid_after_ln = True
    if ret.input_resolution != input_resolution:
        if encoder_type in ['CLIPViT_B_16', 'CLIPViT_L_14']:
            pos = ret.positional_embedding
            patch_size = ret.conv1.kernel_size[0]
        else:
            pos = ret.attnpool.positional_embedding
            patch_size = 32
        p2 = resize_2d_pos_embed(pos,
                                 ret.input_resolution,
                                 patch_size,
                                 input_resolution)
        ret.input_resolution = input_resolution
        if encoder_type in ['CLIPViT_B_16', 'CLIPViT_L_14']:
            ret.positional_embedding = torch.nn.Parameter(p2)
        else:
            ret.attnpool.positional_embedding = torch.nn.Parameter(p2)
    return ret
