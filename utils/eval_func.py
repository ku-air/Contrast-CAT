import torch
import torch.nn.functional as F
import numpy as np
import math

special_tokens = {"[CLS]", "[SEP]"}
special_idxs = {101,102}    
mask = "[PAD]"
mask_id = 0

def predict(model, text_ids, target, att_mask=None, seg_ids=None):
    out = model(text_ids, attention_mask=att_mask, token_type_ids=seg_ids)
    prob = out[0]
    pred_class = torch.argmax(prob, axis=1).cpu().detach().numpy()
    pred_class_prob = F.softmax(prob, dim=1).cpu().detach().numpy()
    
    return pred_class[0], pred_class_prob[:, target][0]

def truncate_words(sorted_idx, text_words, text_ids, replaced_num, seg_ids=None):
    to_be_replaced_idx = []
    i= 0
    while len(to_be_replaced_idx) < replaced_num and i!=len(text_words)-1:
        current_idx = sorted_idx[i]
        if text_words[current_idx] not in special_tokens:
            to_be_replaced_idx.append(current_idx)
        i += 1
    remaining_idx = sorted(list(set(sorted_idx) - set(to_be_replaced_idx)))
    truncated_text_ids = text_ids[0, np.array(remaining_idx)]
    if seg_ids is not None:
        seg_ids = seg_ids[0, np.array(remaining_idx)]
    truncated_text_words = np.array(text_words)[remaining_idx]
    return truncated_text_ids.unsqueeze(0), truncated_text_words, seg_ids

def cal_aopc(original_probs, degradation_probs):
    
    original_probs = np.array(original_probs)
    degradation_probs = np.array(degradation_probs)
    
    result = []
    for i in range(len(original_probs)):
        diffs_k = []
        for j in range(9):            
            diff = original_probs[i] - degradation_probs[i][j]
            diffs_k.append(diff)
        result.append(diffs_k)
    
    return result

def cal_logodds(original_probs, degradation_probs):
    
    original_probs = np.array(original_probs)
    degradation_probs = np.array(degradation_probs)
    
    result = []
    for i in range(len(original_probs)):
        ratios_k = []
        for j in range(9):
            ratio = math.log(degradation_probs[i][j] / original_probs[i])
            ratios_k.append(ratio)
        result.append(ratios_k)
    
    return result