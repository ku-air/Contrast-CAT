import os
import gc
import pickle
import argparse

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
from tqdm import tqdm
import random

############################################################
# Hugging Face datasets and tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer
############################################################

############################################################
from utils.commons import *
import utils.eval_func as custom_eval
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
############################################################

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():

    model_path = args.model_path
    dataset_name = args.dataset_name.lower()

    # Set default model path if not specified
    if dataset_name == 'amazon':

        if model_path == None:
            model_path = "fabriceyhc/bert-base-uncased-amazon_polarity"
            
    elif dataset_name == 'yelp':

        if model_path == None:
            model_path = "fabriceyhc/bert-base-uncased-yelp_polarity"
        
    elif dataset_name == 'sst2':
        
        if model_path == None:
            model_path = "textattack/bert-base-uncased-SST-2"

    elif dataset_name == 'imdb':

        if model_path == None:
            model_path = "fabriceyhc/bert-base-uncased-imdb"
        
    else:
        
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    seed_everything(ran_seed)

    # Load precomputed attribution results
    all_attributes = torch.load(f"./saved_attributions/{dataset_name}/all_attributes_{args.method}.pt")
    degrade_step = 10
    seg_ids = None
    
    original_probs = []
    degradation_probs = []

    # Evaluate each instance
    for instance in tqdm(all_attributes):
        
        text = instance["text"]
        text_words = instance["tokens"]
        pred_class = instance["pred_class"]
        attribute = instance["attribution"].cpu().detach().numpy()
    
        if len(text_words) < 10:
            continue

        # Re-encode the input
        text_ids, att_mask, _ = preprocess_sample(text, tokenizer, device="cuda")

        # Get original prediction probability
        result = model(text_ids)
        prob = F.softmax(result[0], dim=1).cpu().detach().numpy()
        original_prob = prob[:, pred_class][0]

        # Sort tokens by attribution score (descending)
        sorted_idx = np.argsort(-attribute)

        # Define truncation points
        total_len = len(text_words)
        granularity = np.linspace(0, 1, degrade_step)
        trunc_words_num = [int(g) for g in np.round(granularity * total_len)]
        trunc_words_num = list(dict.fromkeys(trunc_words_num))

        # Perform progressive truncation
        instance_probs = []
        for num in trunc_words_num[1:]:
            truncated_text_ids, _, _ = custom_eval.truncate_words(sorted_idx, text_words, text_ids, num)
            trunc_class, trunc_prob = custom_eval.predict(model, truncated_text_ids.to("cuda"), pred_class)
            instance_probs.append(trunc_prob)
    
        original_probs.append(original_prob)
        degradation_probs.append(instance_probs)
    
    # ==============================
    # Compute final evaluation metrics
    # ==============================
    aopc = custom_eval.cal_aopc(original_probs, degradation_probs)
    sample_mean_aopc = np.array(aopc).mean(axis=0)
    
    logodds = custom_eval.cal_logodds(original_probs, degradation_probs)
    sample_mean_logodds = np.array(logodds).mean(axis=0)
    

    # ==============================
    # Save results
    # ==============================
    dir_path = f"./stats/{dataset_name}/"
    os.makedirs(dir_path, exist_ok=True)
    save_path = os.path.join(dir_path, f"eval_results_{args.method}.pickle")

    with open(save_path, 'wb') as f:
        pickle.dump({
            "aopc": sample_mean_aopc,
            "logodds": sample_mean_logodds
        }, f)

##########################################################################################
# ==============================
# Argument parser setup
# ==============================
# Example usage:
# python eval_attribute.py --method contrastcat --dataset_name sst2

parser = argparse.ArgumentParser()

parser.add_argument("--method", type=str, default='contrastcat', help="Attribution Method: contrastcat, attcat, cat, LRP, rollout, gradsam, etc.")
parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name: amazon, yelp, sst2, imdb")
parser.add_argument("--model_path", type=str, default= None, help="Huggingface model path")
args = parser.parse_args()
##########################################################################################

if __name__ == "__main__":
    
    ran_seed = 41
    seed_everything(ran_seed)
    
    main()
    gc.collect()  # optional cleanup