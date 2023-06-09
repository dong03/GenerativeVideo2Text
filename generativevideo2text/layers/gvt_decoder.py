import warnings
from torch.nn import functional as F
import torch
import logging
from torch import nn
from pprint import pformat
import functools
import random
import pdb
from .bert import BertConfig
from .bert.modeling_bert import BertEncoder
from .bert.xbert import BertEncoder as BertCrossAttEncoder
from .predictor import TextGenerator
from .decoder import TransformerDecoderTextualHead, CaptioningModel, convert2valid, ClassificationHead, create_decoder, create_projecton_layer


class BertEncoderAsDecoder(nn.Module):
    def __init__(self, encoder, clf_flag=False):
        super().__init__()
        self.encoder = encoder
        self.clf_flag = clf_flag

    def forward(self, tgt, memory,
                tgt_mask=None,
                # memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_bi_valid_mask=None,
                encoder_history_states=None,
                # tgt_bi_valid_mask: N x num_tgt
                ):
        assert tgt_key_padding_mask is None, 'not supported'
        assert tgt_mask.dim() == 2
        assert tgt_mask.shape[0] == tgt_mask.shape[1]
        # tgt_mask should always be 0/negative infinity
        # mask
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)

        hidden_states = torch.cat((memory, tgt), dim=1)
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory),
                               device=device, dtype=dtype)
        top_right = torch.full((num_memory, num_tgt), float(
            '-inf'), device=tgt.device, dtype=dtype,)
        # top_right[0] = torch.zeros(num_tgt, device=device, dtype=dtype)

        bottom_left = torch.zeros(
            (num_tgt, num_memory), dtype=dtype, device=tgt_mask.device,)

        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

        full_attention_mask = torch.cat((left, right), dim=1)[None, :]

        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full(
                (memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        # if it is False, it means valid. That is, it is not a padding
        assert memory_key_padding_mask.dtype == torch.bool
        zero_negative_infinity = torch.zeros_like(
            memory_key_padding_mask, dtype=tgt.dtype)
        zero_negative_infinity[memory_key_padding_mask] = float('-inf')
        full_attention_mask = full_attention_mask.expand(
            (memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + num_tgt))
        full_attention_mask = full_attention_mask.clone()
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update

        if tgt_bi_valid_mask is not None:
            # verify the correctness
            bs = full_attention_mask.shape[0]
            # during inference, tgt_bi_valid_mask's length is not changed, but
            # num_tgt can be increased
            max_valid_target = tgt_bi_valid_mask.shape[1]
            mask = tgt_bi_valid_mask[:, None, :].expand(
                (bs, num_memory+num_tgt, max_valid_target))
            full_attention_mask[:, :, num_memory:(
                num_memory+max_valid_target)][mask] = 0

        # add axis for multi-head
        full_attention_mask = full_attention_mask[:, None, :, :]
        if encoder_history_states is None:
            result = self.encoder(
                hidden_states=hidden_states,
                attention_mask=full_attention_mask,
                encoder_history_states=encoder_history_states,
            )
            result = list(result)
            cls_pos = result[0][:, -1]
            result[0] = result[0][:, num_memory:].transpose(0, 1)
            if self.encoder.output_hidden_states:
                return result[0], result[1]
            else:
                if self.clf_flag:
                    return result[0], cls_pos
                else:
                    # make it back-compatible
                    return result[0]
        else:
            encoder_out = self.encoder(
                hidden_states=hidden_states[:, -1:],
                attention_mask=full_attention_mask[:, :, -1:],
                encoder_history_states=encoder_history_states,
            )
            result = encoder_out[0].transpose(0, 1)
            if self.encoder.output_hidden_states:
                return result, encoder_out[1]
            else:
                return result


def create_decoder(decoder_type, norm_type,
                   textual_feature_size,
                   attention_heads,
                   feedforward_size,
                   dropout,
                   num_layers,
                   output_hidden_states=False,
                   use_mlp_wrapper=None,
                   vocab_size=21128,
                   cls_flag=False
                   ):
    assert norm_type in ['post', 'pre']
    if decoder_type is None:
        assert NotImplemented
    elif decoder_type == 'bert_en':
        config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=textual_feature_size,
            num_hidden_layers=num_layers,
            num_attention_heads=attention_heads,
            intermediate_size=feedforward_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
        )
        config.pre_norm = (norm_type == 'pre')
        config.use_mlp_wrapper = use_mlp_wrapper
        config.output_hidden_states = output_hidden_states
        encoder = BertEncoder(config)
        return BertEncoderAsDecoder(encoder, cls_flag)


class TransformerDecoderClfTextualHead(TransformerDecoderTextualHead):
    def __init__(self, visual_feature_size: int, vocab_size: int, hidden_size: int, num_layers: int, attention_heads: int, feedforward_size: int, dropout: float = 0.1, norm_type: str = "post", mask_future_positions: bool = True, max_caption_length: int = 30, padding_idx: int = 0, decoder_type=None, visual_projection_type=None, not_tie_weight=None, output_hidden_states=None, use_mlp_wrapper=None, cosine_linear=False, ):
        super().__init__(visual_feature_size, vocab_size, hidden_size, num_layers, attention_heads, feedforward_size, dropout, norm_type, mask_future_positions,
                         max_caption_length, padding_idx, decoder_type, visual_projection_type, not_tie_weight, output_hidden_states, use_mlp_wrapper, cosine_linear)

        self.classifier = ClassificationHead(hidden_size=768)
        self.transformer = create_decoder(
            decoder_type=decoder_type,
            norm_type=norm_type,
            textual_feature_size=self.textual_feature_size,
            attention_heads=self.attention_heads,
            feedforward_size=self.feedforward_size,
            dropout=dropout,
            num_layers=self.num_layers,
            output_hidden_states=output_hidden_states,
            use_mlp_wrapper=use_mlp_wrapper,
            vocab_size=vocab_size,
            cls_flag=True
        )

        self.text_projection = create_projecton_layer(
            None, self.textual_feature_size, self.textual_feature_size)
        '''
        config = self.transformer.encoder.config
        config.num_hidden_layers = 2
        config.fusion_layer = 0
        config.chunk_size_feed_forward = 0
        config.encoder_width = 768
        self.crossattn = BertCrossAttEncoder(config)
        '''
        self.apply(self._init_weights)

    def forward(
        self,
        hidden_states,
        caption_tokens,
        hidden_valid_mask=None,  # can be None
        caption_lengths=None,  # useless
        bi_valid_mask_caption=None,
        # caption_mask=None,
        encoder_history_states=None,
        return_dict=False,
        text_feature=None
    ):
        if return_dict:
            ret = {}

        projected_visual_features = self.visual_projection(
            hidden_states) if hidden_states is not None else None
        if return_dict:
            ret['projected_visual_features'] = projected_visual_features
        batch_size, max_caption_length = caption_tokens.size()

        if text_feature is None:
            caption_embeddings = self.embedding(caption_tokens)

            # An additive mask for masking the future (one direction).

            # We transpose the first two dimensions of tokens embeddings and visual
            # features, as required by decoder.
            caption_embeddings = caption_embeddings.transpose(0, 1)
        else:
            caption_embeddings = self.text_projection(
                text_feature).transpose(0, 1)

        uni_mask_zero_neg = self._generate_future_mask(
            max_caption_length, caption_embeddings.dtype, caption_embeddings.device
        )

        if projected_visual_features is not None:
            projected_visual_features = projected_visual_features.transpose(
                0, 1)
        else:
            projected_visual_features = torch.zeros(
                (0, caption_embeddings.shape[1], caption_embeddings.shape[2]),
                dtype=caption_embeddings.dtype,
                device=caption_embeddings.device,
            )

        extra_param = {}
        if bi_valid_mask_caption is not None:
            extra_param = {'tgt_bi_valid_mask': bi_valid_mask_caption}
        if not isinstance(self.transformer, torch.nn.modules.transformer.TransformerDecoder):
            extra_param['encoder_history_states'] = encoder_history_states

        # if transformer here is the pytorch/decoder, there is no chance, the
        # output is always tensor
        trans_out, cls_embedding = self.transformer(
            caption_embeddings,
            projected_visual_features,
            memory_key_padding_mask=(hidden_valid_mask.logical_not(
            ) if hidden_valid_mask is not None else None),
            tgt_mask=uni_mask_zero_neg,
            # tgt_key_padding_mask=caption_mask,
            # encoder_history_states=encoder_history_states,
            **extra_param,
        )
        if isinstance(trans_out, tuple):
            textual_features = trans_out[0]
        else:
            assert isinstance(trans_out, torch.Tensor)
            textual_features = trans_out
        # Undo the transpose and bring batch to dim 0.
        # shape: (batch_size, max_caption_length, hidden_size)
        textual_features = textual_features.transpose(0, 1)
        if return_dict:
            ret['textual_features'] = textual_features

        # shape: (batch_size, max_caption_length, vocab_size)
        # import pdb
        # pdb.set_trace()
        '''
        cls_embedding = self.crossattn(
            hidden_states=trans_out.permute(1,0,2),
            attention_mask = uni_mask_zero_neg,
            encoder_hidden_states=projected_visual_features[::197].permute(1,0,2)
        )
        cls_embedding = cls_embedding.last_hidden_state[:, 0]
        # cls_embedding = torch.cat(
        #     [cls_embedding, torch.mean(projected_visual_features[::197].permute((1, 0, 2)), dim=1)], dim=1)
        # cls_embedding = textual_features[:, -1]
        '''
        cls_prob = self.classifier(cls_embedding)
        output_logits = self.output(textual_features)
        if isinstance(trans_out, tuple):
            if return_dict:
                ret['output_logits'] = output_logits
                ret['history'] = trans_out[1]
                ret['cls_prob'] = cls_prob
                return ret
            else:
                return output_logits, trans_out[1], cls_prob
        else:
            if return_dict:
                ret['output_logits'] = output_logits
                ret['cls_prob'] = cls_prob
                return ret
            else:
                return output_logits, cls_prob


class CaptioningGVTModel(CaptioningModel):
    def __init__(self, visual, textual, sos_index=1, eos_index=2, decoder=None, loss_type=None, context_not_share_embedding=False, scst=False, tokenizer=None, scst_temperature=1, use_history_for_infer=False, pooling_images=None, num_image_with_embedding=0, text_encoder=None):
        super().__init__(visual, textual, sos_index, eos_index, decoder, loss_type, context_not_share_embedding,
                         scst, tokenizer, scst_temperature, use_history_for_infer, pooling_images, num_image_with_embedding)

        self.text_encoder = text_encoder
        pass

    def forward_one(self, batch, return_info=False):
        # shape: (batch_size, max_caption_length, vocab_size)
        if 'image' in batch:
            #BZ * frame * 3 * 160 * 160
            try:
                input = batch['image'].reshape(-1, batch['image'].shape[-3],
                                               batch['image'].shape[-2], batch['image'].shape[-1])
                features = self.image_encoder(input)
                bz, num_frame = batch['image'].shape[:2]

                corse_features = features[:, 0, :].reshape(
                    batch['image'].shape[0], batch['image'].shape[1], -1)
                fine_features = features[[num_frame //
                                          2 + num_frame * i for i in range(bz)]]

                features = torch.cat([corse_features, fine_features], dim=1)
                # features = features.reshape(
                #     batch['image'].shape[0], batch['image'].shape[1], features.shape[-2], features.shape[-1])
                # if self.num_image_with_embedding:
                #     for i in range(self.num_image_with_embedding):
                #         features[:, i] += self.img_temperal_embedding[i]

                if self.pooling_images == 'avg':
                    visual_features = torch.mean(features, dim=1)
                elif self.pooling_images is None:
                    visual_features = features.reshape(
                        features.shape[0], -1, 768)
            except:
                if isinstance(batch['image'], (list, tuple)):
                    features = [self.image_encoder(im)
                                for im in batch['image']]
                    if self.num_image_with_embedding:
                        features = [
                            f + e for f, e in zip(features, self.img_temperal_embedding)]
                    if self.pooling_images is None:
                        visual_features = torch.cat(features, dim=1)
                    elif self.pooling_images == 'avg':
                        visual_features = torch.stack(
                            features, dim=1).mean(dim=1)
                    else:
                        raise NotImplementedError
                else:
                    visual_features = self.image_encoder(batch['image'])
        else:
            visual_features = None
        visual_features_valid = None
        if 'context' in batch:
            context_embedding = self.context_embedding if self.context_not_share_embedding else self.textual.embedding
            all_context = [visual_features]
            all_valid = [convert2valid(visual_features.shape[:2])]
            for info in batch['context']:
                context = context_embedding(info['tokens'])
                valid = convert2valid(info['tokens'].shape, info['length'])
                all_context.append(context)
                all_valid.append(valid)
            visual_features = torch.cat(all_context, dim=1)
            visual_features_valid = torch.cat(all_valid, dim=1)
        if not self.training or (not self.scst):
            return self.forward_one_ce(batch, visual_features, visual_features_valid, return_info)
        else:
            assert self.training and self.scst
            return self.forward_one_scst(batch, visual_features, visual_features_valid)

    @torch.no_grad()
    def infer(self, batch, visual_features, visual_features_valid,
              search_param=None):
        batch_size = visual_features.size(0)
        if 'prefix' not in batch:
            start_predictions = visual_features.new_full(
                (batch_size, 1), self.sos_index
            ).long()
        else:
            # if batch size is larger than 1, the prefix length could be
            # different, and we have to padding non-valid data, which
            # is not supported
            assert len(batch['prefix']) == 1, 'not supported'
            start_predictions = batch['prefix'].long()

        self.prev_encoded_layers = None
        # Add image features as a default argument to match callable
        # signature accepted by beam search class (partial captions only).
        '''
        padding_mask = torch.ones_like(
            visual_features[:, :, 0]).long()

        predictor_inputs = [visual_features, padding_mask, start_predictions]
        topk_ids, topk_probs = self.beam_generator.translate_batch_scst(
            predictor_inputs, out_size=1)
        output_dict = {
            'predictions': topk_ids,
            'logprobs': topk_probs,
        }
        return output_dict
        '''

        decoding_step = functools.partial(
            self.decoding_step, visual_features, visual_features_valid,
            batch.get('bi_valid_mask_caption')
        )

        search_param = search_param or {}
        # the start_predictions are not in predicted_caption
        predicted_caption, logprobs = self.decoder.search(
            start_predictions, decoding_step, **search_param
        )
        # pdb.set_trace()
        if 'prefix' in batch:
            # we need to remove prefix from predicted_caption
            predicted_caption = predicted_caption[:,
                                                  start_predictions.shape[1]:]
        output_dict = {
            'predictions': predicted_caption,
            'logprobs': logprobs,
        }
        return output_dict
