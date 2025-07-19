import os
import gc
import pickle
import argparse

import torch
from scipy.special import softmax
############################################################
# Hugging Face datasets and tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer
############################################################

############################################################
from utils.commons import preprocess_sample
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
############################################################

def main():

    # Select model and dataset based on dataset name
    if args.dataset_name.lower() == 'amazon':
        
        model_path = "fabriceyhc/bert-base-uncased-amazon_polarity"
        dataset = load_dataset("amazon_polarity", split='train', streaming=True) # load dataset

    elif args.dataset_name.lower() == 'yelp':
        
        model_path = "fabriceyhc/bert-base-uncased-yelp_polarity"
        dataset = load_dataset("yelp_polarity", split='train', streaming=True) # load dataset
        
    elif args.dataset_name.lower() == 'sst2':
        
        model_path = "textattack/bert-base-uncased-SST-2"
        dataset = load_dataset("sst2", split='train', streaming=True) # load dataset
        
    elif args.dataset_name.lower() == 'imdb':
        
        model_path = "fabriceyhc/bert-base-uncased-imdb"
        dataset = load_dataset("imdb", split='train', streaming=True) # load dataset
        
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Initialize activation library for both target classes
    library = [[],[]] # library[0] → target class 1 examples, library[1] → target class 0 examples
    for i, test_instance in enumerate(dataset):

        if args.dataset_name.lower() == 'amazon':
            text = test_instance['content']
            
        elif args.dataset_name.lower() == 'yelp':
            text = test_instance['text']
            
        elif args.dataset_name.lower() == 'imdb':
            text = test_instance['text']
            
        elif args.dataset_name.lower() == 'sst2':
            text = test_instance['sentence']
        
        target = test_instance['label'] 

        # Preprocess the input (returns token ids, attention mask, and tokens)
        text_ids, att_mask, text_words = preprocess_sample(text, tokenizer, device)
    
        # Skip examples with too few tokens
        if len(text_words)< 10: 
            continue

        # Forward pass through model (no attention mask used here)
        result = model(input_ids = text_ids, attention_mask = None, output_hidden_states=True)
        prob = result[0] # Logits
        hs = result[1] # Hidden states

        # Predicted class and softmax confidence scores
        pred_class = torch.argmax(prob, axis=1).cpu().detach().numpy().squeeze()
        pred_class_prob = softmax(prob.cpu().detach().numpy(), axis=1).squeeze()
        #pred_class_prob = softmax(prob, dim=1).squeeze().cpu().numpy()

        # Skip misclassified examples
        if target != pred_class.item() :
            continue

        # Determine the opposite class (target for misclassification)
        lib_target = 1 if target == 0 else 0

         # Store hidden states on CPU for the opposite class
        cpu_hs = tuple(h.detach().cpu() for h in hs)
        temp_lib_dict = {'target_cls_confi':pred_class_prob[lib_target], 'activation':cpu_hs}

        # Append to the corresponding class's library and sort by confidence (ascending)
        library[lib_target].append(temp_lib_dict)
        library[lib_target].sort(key=lambda x: x['target_cls_confi'],reverse=False)

        # Keep only top 50 examples with lowest opposite-class confidence
        if len(library[lib_target]) > 50:
            
            library[lib_target].pop()
            
            break

    # Save the activation libraries as a NumPy file
    save_path = './ref_lib/{}.pkl'.format(args.dataset_name.lower())
    os.makedirs('./ref_lib', exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(library, f)

##########################################################################################
# ==============================
# Argument parser setup
# ==============================
parser = argparse.ArgumentParser()

# Required dataset name argument
parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name: amazon, yelp, sst2, imdb")
args = parser.parse_args()
##########################################################################################

if __name__ == "__main__":
    main()
    gc.collect()  # optional cleanup