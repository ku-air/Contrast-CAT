import os
import gc
import pickle
import argparse

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
import random
############################################################
# Hugging Face datasets and tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer
############################################################

############################################################
from utils.commons import *
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
    reference_library_path = args.reference_library_path

    # Set default model path if not provided
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

    # Load reference library if needed
    library = None
    if args.method == 'contrastcat':
        library = np.load(reference_library_path, allow_pickle=True)

    # initialize the explanations generator
    attribute_generator = Attribute_Generator(model,library)
    
    seed_everything(ran_seed)
    to_test = np.array(dataset['validation'])

    all_attributes = []
    for i, test_instance in enumerate(to_test):

        # Get the input text depending on dataset format
        if dataset_name == 'sst2':
            
            text = test_instance['sentence']
            
        elif dataset_name in ['yelp', 'imdb']:
            
            text = test_instance['text']
            
        elif dataset_name == 'amazon':
            
            text = test_instance['content']
            
        else:
            
            raise ValueError(f"Invalid dataset name: {dataset_name}")
            
        target = test_instance['label'] 

        # Tokenize and preprocess input
        text_ids, att_mask, text_words = preprocess_sample(text, tokenizer, device)
        
        # Skip if input too short
        total_len = len(text_words)
        if total_len< 10: 
            continue

        # Run model to get prediction
        result = model(text_ids, attention_mask=None, token_type_ids=None)
        prob = result[0]
    
        pred_class_prob = F.softmax(prob, dim=1).cpu().detach().numpy()
        pred_class = torch.argmax(prob, axis=1).cpu().detach().numpy()[0]
        original_prob = pred_class_prob[:, pred_class][0]
    
        #################################################################################
        # Compute attribution
        attribute = attribute_generator.get_attribution(
            args.method,
            input_ids=text_ids,
            attention_mask=att_mask,
            index=pred_class,
            start_layer=0,
            text_words=text_words,
            text_ids=text_ids,
            original_prob=original_prob
        )

        # Save the attribution with metadata
        all_attributes.append({
            "text": text,
            "tokens": text_words,
            "pred_class": pred_class,
            "attribution": attribute.cpu()
        })

    # Save all attribution results to disk
    save_path = f"./saved_attributions/{dataset_name}/"
    os.makedirs(save_path, exist_ok=True)
    torch.save(all_attributes, os.path.join(save_path, f"all_attributes_{args.method}.pt"))

##########################################################################################
# ==============================
# Argument parser setup
# ==============================
# Example usage:
# python get_attribute.py --method contrastcat --dataset_name sst2 --reference_library_path './ref_lib/sst.npy'

parser = argparse.ArgumentParser()

parser.add_argument("--method", type=str, default='contrastcat', help="Attribution Method: contrastcat, attcat, cat, LRP, rollout, gradsam, etc.")
parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name: amazon, yelp, sst2, imdb")
parser.add_argument("--model_path", type=str, default= None, help="Huggingface model path")
parser.add_argument("--reference_library_path", type=str, default= './ref_lib/amazon.pkl', help="Path to reference library .pkl file")
args = parser.parse_args()
##########################################################################################

if __name__ == "__main__":
    
    ran_seed = 41
    seed_everything(ran_seed)
    
    main()
    gc.collect()  # optional cleanup