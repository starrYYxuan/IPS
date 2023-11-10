from typing import Optional
import torch
import numpy
import torch.nn.functional as F
from transformers.generation_utils import GenerationMixin
from processor import Processor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import json
import os
from torch.nn.utils.rnn import pad_sequence


def encdec_enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            # item = item.expand(bsz*beam_width, -1, -1, -1).contiguous()    # [bsz*beam, num_head, seq_len, esz]
            # item = item.expand(beam_width, -1, -1, -1, -1).reshape(bsz * beam_width, num_head, seq_len,
            #                                                       esz).contiguous()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz * beam_width, num_head, seq_len,
                                                                                esz).contiguous()
            items.append(item)
        new_key_values.append(tuple(items))
    return tuple(new_key_values)


def encdec_select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam // beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0)).contiguous()  # [B, K, num_head, seq_len, esz]
            item = item[range(bsz), selected_idx, :, :, :]  # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


def ranking_fast(
        context_hidden,
        next_hidden,
        next_top_k_probs,
        eos_hidden,
        alpha,
        beam_width,
        eos_attention_mask
):
    """
        context_hidden: bsz*beam x seqlen x embed_dim
        context_hidden can be added on seqlen dimension so we can get the response latent variable.
        eos_hidden N x 1 x embed_dim，N is the number of past sentences we choose
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
        next_top_k_probs: the conditional probabilities calculated by LM
    """

    # token level contrastive search.
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=-1, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=-1, keepdim=True)
    token_cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(
        -1)  # [B*K, S]
    token_cosine_similarity = torch.sum(token_cosine_matrix, dim=-1)  # dim=0, [30]
    token_cosine_similarity = torch.div(token_cosine_similarity, context_len)  #

    # utterance level contrastive search.
    # print("eos_hidden_size",eos_hidden.size())

    mid_context_hidden = torch.cat([context_hidden, next_hidden], dim=1)

    # _, utterance_len, _ = eos_hidden.size()
    bsz, eos_num, esz = eos_hidden.size()
    eos_hidden = eos_hidden.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz * beam_width, eos_num, esz)
    norm_eos_hidden = eos_hidden / eos_hidden.norm(dim=-1, keepdim=True)
    response_hidden = torch.mean(mid_context_hidden, dim=1).unsqueeze(1)  # torch.max()
    norm_response_hidden = response_hidden / response_hidden.norm(dim=1, keepdim=True)
    utterance_cosine_matrix = torch.matmul(norm_response_hidden, norm_eos_hidden.transpose(1, 2)).squeeze(1)
    eos_attention_mask = eos_attention_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(bsz * beam_width, eos_num)
    utterance_cosine_matrix_mask = utterance_cosine_matrix.mul(eos_attention_mask)
    utterance_cosine_similarity = torch.sum(utterance_cosine_matrix_mask, dim=-1)

    utterance_len = []
    for x, each_utterance in enumerate(eos_attention_mask):
        eos_num = 0
        for i in range(len(each_utterance)):
            if each_utterance[i] != 0:
                eos_num += 1
            else:
                break
        utterance_len.append((eos_num))

    utterance_len = torch.tensor(utterance_len)
    utterance_cosine_similarity_mean = torch.div(utterance_cosine_similarity, utterance_len)

    # scores, _ = torch.max((token_cosine_similarity - utterance_cosine_similarity_mean), dim=-1)  #
    scores = token_cosine_similarity - utterance_cosine_similarity_mean
    # scores = 0.6 * token_cosine_similarity - 0.4 * utterance_cosine_similarity_mean
    next_top_k_probs = next_top_k_probs.view(-1)  # [B*K]
    scores = alpha * next_top_k_probs + (1 - alpha) * scores
    scores = torch.stack(torch.split(scores, beam_width))  # [B, K]
    # selected_idx = scores.max(dim=-1)[1]  # [B]

    # scores = torch.where(torch.isinf(scores), torch.full_like(scores, 0), scores)
    # print(scores)
    scores = torch.nn.functional.softmax(scores)
    scores = torch.tensor(scores)
    selected_idx = torch.multinomial(scores, 1)
    selected_idx = selected_idx.squeeze(1)

    return selected_idx


def hierarchically_contrastive_one_step(
        model,
        beam_width,
        alpha,
        last_hidden_states,
        eos_hidden,
        logit_for_next_step,
        past_key_values,
        eos_attention_mask,
        encoder_outputs,
        decoder_ids,
        logits_processor,
        attention_mask,
):
    # input_ids: [B, S]
    bsz, seqlen, embed_dim = last_hidden_states.size()

    # next_probs = F.softmax(logit_for_next_step, dim=-1)
    # _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)  # [B, K] [1, 1]
    # top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)

    logits_processor = logits_processor
    _, input_len = attention_mask.size()
    attention_mask = attention_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(bsz * beam_width, input_len,
                                                                                    )

    next_tokens_scores = logits_processor(decoder_ids, logit_for_next_step)
    _, next_token_ids = torch.topk(next_tokens_scores, dim=-1, k=beam_width)
    top_k_probs = torch.gather(next_tokens_scores, dim=1, index=next_token_ids)

    past_key_values = encdec_enlarge_past_key_values(past_key_values, beam_width)

    output = model(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        decoder_input_ids=next_token_ids.contiguous().view(-1, 1),
        # decoder_input_ids=top_k_ids.contiguous().view(-1, 1),
        output_hidden_states=True,
        past_key_values=past_key_values,
        use_cache=True,
    )

    past_key_values = output.past_key_values

    logits = output.logits[:, -1, :]  # [B*K, V]
    next_hidden = output.decoder_hidden_states[-1]  # [B*K, 1, E] [1, 1, 1024]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz * beam_width, seqlen,
                                                                                            embed_dim)  # [B*K, S, E], [1, 30, 1024]

    selected_idx = ranking_fast(
        context_hidden,
        next_hidden,
        top_k_probs,
        eos_hidden,
        alpha,
        beam_width,
        eos_attention_mask,
    )  # [B]

    # next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)  # [B, 1]
    next_id = next_token_ids[range(len(next_token_ids)), selected_idx].unsqueeze(-1)
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))  # [B, K, E]
    next_hidden = next_hidden[range(bsz), selected_idx, :]  # [B, E]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)  # [B, S, E]
    past_key_values = encdec_select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]  # [B, V]

    return next_id, last_hidden_states, logits, encoder_outputs, eos_hidden, past_key_values


def hierarchically_decoding(
        attention_mask: Optional[torch.LongTensor],
        eos_position: Optional[torch.LongTensor],
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[torch.nn.Module] = None,
        input_ids: Optional[torch.LongTensor] = None,
        decoder_ids: Optional[torch.LongTensor] = None,
        beam_width: Optional[int] = None,
        alpha=None,
        min_decoding_length: Optional[int] = None,
        max_decoding_length: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        first_k_steps: Optional[int] = None,
        early_stop: Optional[bool] = False,
        hcd_padding_value: Optional[int] = None,
        **kwargs
):
    # first stage: greedy search, beam search, top-k,top-p
    sampled_output = model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
                                    max_length=first_k_steps+1,
                                    min_length=min_decoding_length,
                                    no_repeat_ngram_size=2,
                                    early_stopping=True,
                                    num_beams=kwargs["num_beams"] if kwargs["num_beams"] is not None else None,
                                    # num_beams=1,
                                    top_p=kwargs["top_p"] if kwargs["top_p"] is not None else None,
                                    top_k=kwargs["top_k"] if kwargs["top_k"] is not None else None,
                                    do_sample=True,
                                    )

    sampled_output = sampled_output.numpy().tolist()
    for i in range(len(sampled_output)):
        batch_len = len(sampled_output[i])
        while batch_len:
            if sampled_output[i][batch_len - 1] == model.config.decoder_start_token_id:
                sampled_output[i].pop(-1)
                batch_len -= 1
            else:
                break

    for i in range(len(sampled_output)):
        sampled_output[i] = torch.tensor(sampled_output[i])

    # sampled_output = pad_sequence(sampled_output, batch_first=True, padding_value=model.config.pad_token_id)
    sampled_output = pad_sequence(sampled_output, batch_first=True, padding_value=hcd_padding_value)
    decoder_ids = torch.tensor(sampled_output).long()

    # decoder_attention_mask = (sampled_output != model.config.pad_token_id).float()
    decoder_attention_mask = (sampled_output != hcd_padding_value).float()

    if input_ids.is_cuda:
        decoder_ids = decoder_ids.cuda(input_ids.get_device())
        decoder_attention_mask = decoder_attention_mask.cuda(input_ids.device())

    # then do contrastive search
    assert 0. <= alpha <= 1.0

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_ids,
        decoder_attention_mask=decoder_attention_mask,
        output_hidden_states=True,
        output_attentions=True,
    )

    contrastive_step = max_decoding_length - first_k_steps
    logits_processor = model._get_logits_processor(
        no_repeat_ngram_size=2,
        min_length=3,
        max_length=contrastive_step,
        # max_length=7,
        forced_bos_token_id=tokenizer.bos_token_id,
        forced_eos_token_id=eos_token_id,
        repetition_penalty=None,
        encoder_no_repeat_ngram_size=None,
        encoder_input_ids=input_ids,
        bad_words_ids=None,
        eos_token_id=eos_token_id,
        prefix_allowed_tokens_fn=None,
        num_beams=kwargs["num_beams"] if kwargs["num_beams"] is not None else 1,
        num_beam_groups=None,
        diversity_penalty=None,
        remove_invalid_values=None,
    )
    past_key_values = output.past_key_values

    decoder_hidden_states = output.decoder_hidden_states,
    last_hidden_states = output.decoder_hidden_states[-1]  # [B, S, E]
    # last_hidden_states = sampled_outputs.decoder_hidden_states[-2]

    logit_for_next_step = output.logits[:, -1, :]  # [B, V] [3, 50266]
    # logit_for_next_step = log_probs[-2]

    encoder_outputs = (output.encoder_last_hidden_state, output.encoder_hidden_states, output.encoder_attentions)

    eos_attention_mask = (eos_position != 0).float()
    for i in range(eos_attention_mask.size(0)):
        eos_attention_mask[i][0] = 1
        eos_attention_mask[i][1] = 1

    eos_position = eos_position.unsqueeze(-1).repeat(1, 1, encoder_outputs[0].shape[-1])
    eos_hidden = torch.gather(encoder_outputs[0], dim=1, index=eos_position)


    sampled_output = sampled_output.detach().cpu().tolist()

    for step in range(contrastive_step):
        decoder_ids, last_hidden_states, logit_for_next_step, input_embeds, eos_hidden, past_key_values = hierarchically_contrastive_one_step(
            model,
            beam_width,
            alpha,
            last_hidden_states,
            eos_hidden,
            logit_for_next_step,
            past_key_values,
            eos_attention_mask,
            encoder_outputs,
            decoder_ids,
            logits_processor,
            attention_mask)
        token = decoder_ids.squeeze(-1).tolist()

        for i in range(len(sampled_output)):
            sampled_output[i] = sampled_output[i] + [token[i]]

        decoder_ids = torch.tensor(sampled_output)

    for x, each_output in enumerate(sampled_output):
        for i in range(len(each_output)):
            if each_output[i] == model.config.decoder_start_token_id and i > 0:
                sampled_output[x] = each_output[:i + 1]
                break

    return sampled_output
