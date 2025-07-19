import torch
from torch.nn.functional import softmax
import numpy as np
import torch.nn.functional as F

special_tokens = {"[CLS]", "[SEP]"}
special_idxs = {101,102}    
mask = "[PAD]"
mask_id = 0   

def preprocess_sample(text, tokenizer, device):
    tokenized_input  = tokenizer(text, add_special_tokens=True, truncation=True)
    input_ids = tokenized_input['input_ids']
    text_ids = (torch.tensor([input_ids])).to(device)
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])
    
    # mask special tokens
    att_mask = tokenized_input['attention_mask']
    spe_idxs = [x for x, y in list(enumerate(input_ids)) if y in special_idxs]
    att_mask = [0 if index in spe_idxs else 1 for index, item in enumerate(att_mask)]
    att_mask = [0 if index in spe_idxs else 1 for index, item in enumerate(att_mask)]
    att_mask = (torch.tensor([att_mask])).to(device)
    
    return text_ids, att_mask, text_words
    
class Attribute_Generator:
    
    def __init__(self, model, lib=None):
        
        self.model = model
        self.lib = lib if lib is not None else {}
        
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def get_attribution(self, method_name, **kwargs):
        
        method_name = method_name.lower()

        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        index = kwargs.get("index")
        start_layer = kwargs.get("start_layer", 0)
        
        if method_name == "contrastcat":
            
            return self.generate_ContrastCAT(
                input_ids=input_ids,
                attention_mask=attention_mask,
                index=index,
                start_layer=start_layer,
                text_words=kwargs.get("text_words"),
                text_ids=kwargs.get("text_ids"),
                original_prob=kwargs.get("original_prob")
            )
            
        elif method_name == "attcat":
            
            return self.generate_attcat(
                input_ids=input_ids,
                attention_mask=attention_mask,
                index=index,
                start_layer=start_layer
            )
            
        elif method_name == "cat":
            
            return self.generate_cat(
                input_ids=input_ids,
                attention_mask=attention_mask,
                index=index,
                start_layer=start_layer
            )
            
        elif method_name == "lrp":
            
            return self.generate_LRP(
                input_ids=input_ids,
                attention_mask=attention_mask,
                index=index,
                start_layer=start_layer
            )
            
        elif method_name == "partiallrp":

            return self.generate_LRP_last_layer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                index=index
            )
            
        elif method_name == "attention":
            
            return self.generate_attn_last_layer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                index=index
            )
            
        elif method_name == "rollout":
            
            return self.generate_rollout(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_layer=start_layer,
                index=index
            )
            
        elif method_name == "gradsam":
            
            return self.generate_gradsam(
                input_ids=input_ids,
                attention_mask=attention_mask,
                index=index
            )
            
        else:
            raise ValueError(f"[Attribute_Generator] Unknown attribution method: {method_name}")
            
    def generate_attcat(self, input_ids, attention_mask,
                          index=None, start_layer=0):

        result = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        output = result[0]
        hs = result[1]

        kwargs = {"alpha": 1}

        blocks = self.model.bert.encoder.layer

        for blk_id in range(len(blocks)):
            hs[blk_id].retain_grad()

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        cams = {}
        for blk_id in range(len(blocks)):
            hs_grads = hs[blk_id].grad.detach().cpu()
            
            att = blocks[blk_id].attention.self.get_attn().squeeze(0).detach().cpu()
            att = att.mean(dim=0)
            att = att.mean(dim=0)
            
            cat = (hs_grads * hs[blk_id].detach().cpu()).sum(dim=-1).squeeze(0)
            cat = cat * att
            
            cams[blk_id] = cat
            
        trans_expln = sum(cams.values())
        
        #################################
        ######## Min-Max scaling ########
        min_v = trans_expln.min()
        max_v = trans_expln.max()

        numerator = trans_expln - min_v
        denominator = max_v - min_v

        if denominator == 0:
            trans_expln = numerator/ (denominator + 1e-6)
        else:
            trans_expln = numerator/ denominator
        #################################  
        
        return trans_expln.squeeze(0)
        
    def generate_cat(self, input_ids, attention_mask,
                          index=None, start_layer=0):

        result = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        output = result[0]
        hs = result[1]

        kwargs = {"alpha": 1}

        blocks = self.model.bert.encoder.layer

        for blk_id in range(len(blocks)):
            hs[blk_id].retain_grad()

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        cams = {}
        for blk_id in range(len(blocks)):
            
            hs_grads = hs[blk_id].grad.detach().cpu()
            
            cat = (hs_grads * hs[blk_id].detach().cpu()).sum(dim=-1).squeeze(0).detach().cpu()

            cams[blk_id] = cat
            
        trans_expln = sum(cams.values())
        
        #################################
        ######## Min-Max scaling ########
        min_v = trans_expln.min()
        max_v = trans_expln.max()

        numerator = trans_expln - min_v
        denominator = max_v - min_v

        if denominator == 0:
            trans_expln = numerator/ (denominator + 1e-6)
        else:
            trans_expln = numerator/ denominator
        #################################
        
        return trans_expln.squeeze(0)
    
    def generate_LRP(self, input_ids, attention_mask,
                     index=None, start_layer=11):
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cams = []
        blocks = self.model.bert.encoder.layer
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach().cpu()
            cam = blk.attention.self.get_attn_cam().detach().cpu()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = self.compute_rollout_attention(cams, start_layer=start_layer)
        rollout[:, 0, 0] = 0
        
        trans_expln = rollout[:, 0]
        #################################
        ######## Min-Max scaling ########
        min_v = trans_expln.min()
        max_v = trans_expln.max()

        numerator = trans_expln - min_v
        denominator = max_v - min_v

        if denominator == 0:
            trans_expln = numerator/ (denominator + 1e-6)
        else:
            trans_expln = numerator/ denominator
        #################################        
        
        return trans_expln.squeeze(0)

    def generate_LRP_last_layer(self, input_ids, attention_mask,
                     index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn_cam()[0].detach().cpu()
        cam = cam.clamp(min=0).mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        
        trans_expln = cam[:, 0]
        
        #################################
        ######## Min-Max scaling ########
        min_v = trans_expln.min()
        max_v = trans_expln.max()

        numerator = trans_expln - min_v
        denominator = max_v - min_v

        if denominator == 0:
            trans_expln = numerator/ (denominator + 1e-6)
        else:
            trans_expln = numerator/ denominator
        #################################        

        return trans_expln.squeeze(0)

    def generate_attn_last_layer(self, input_ids, attention_mask, index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn()[0]
        cam = cam.mean(dim=0).unsqueeze(0).detach().cpu()
        cam[:, 0, 0] = 0
        
        trans_expln = cam[:, 0]
        
        #################################
        ######## Min-Max scaling ########
        min_v = trans_expln.min()
        max_v = trans_expln.max()

        numerator = trans_expln - min_v
        denominator = max_v - min_v

        if denominator == 0:
            trans_expln = numerator/ (denominator + 1e-6)
        else:
            trans_expln = numerator/ denominator
        #################################        

        return trans_expln.squeeze(0)

    def generate_rollout(self, input_ids, attention_mask, start_layer=0, index=None):
        self.model.zero_grad()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        blocks = self.model.bert.encoder.layer
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attention.self.get_attn().detach().cpu()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach().cpu()
            all_layer_attentions.append(avg_heads)
        rollout = self.compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        rollout[:, 0, 0] = 0
        
        trans_expln = rollout[:, 0]
        
        #################################
        ######## Min-Max scaling ########
        min_v = trans_expln.min()
        max_v = trans_expln.max()

        numerator = trans_expln - min_v
        denominator = max_v - min_v

        if denominator == 0:
            trans_expln = numerator/ (denominator + 1e-6)
        else:
            trans_expln = numerator/ denominator
        #################################

        return trans_expln.squeeze(0)
        
    def generate_ContrastCAT(self, input_ids, attention_mask,index=None, start_layer=0, text_words=None,text_ids=None, original_prob = None):

        result = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        output = result[0]
        hs = result[1]
        seq_length = hs[0].shape[1]

        kwargs = {"alpha": 1}

        blocks = self.model.bert.encoder.layer

        for blk_id in range(len(blocks)):
            hs[blk_id].retain_grad()

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
        
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        output_cam = []
        
        if len(self.lib[index]) < 30:
            t_len = len(self.lib[index])

        else:
            t_len = 30

        #ATTRIBUTION WITH MULTIPLE CONTRAST
        for reference_hs in self.lib[index][:t_len] :

            cams = {}        
            for blk_id in range(len(blocks)):
                
                hs_grads = hs[blk_id].grad.detach().cpu()

                #################################
                att = blocks[blk_id].attention.self.get_attn().squeeze(0).detach().cpu()
                att = att.mean(dim=0)
                att = att.mean(dim=0)
                #################################

                #################################
                ### Load reference ###
                ref_hs = reference_hs['activation'][blk_id]
                ref_hs_seq_length = ref_hs.shape[1]

                #################################
                if ref_hs_seq_length >= seq_length:
                    reference = ref_hs[:,:seq_length,:]
                else:
                    pad_length = seq_length - ref_hs_seq_length
                    reference = F.pad(input=ref_hs, pad=(0, 0, 0, pad_length), mode='constant', value=0)

                activation = (hs[blk_id].detach().cpu() - reference)

                cat = (hs_grads * activation).sum(dim=-1).squeeze(0)
                cat = cat * att
                #################################
                
                cams[blk_id] = cat

            expln = sum(cams.values())
            
            #################################
            ######## Min-Max scaling ########
            min_v = expln.min()
            max_v = expln.max()
            
            numerator = expln - min_v
            denominator = max_v - min_v

            if denominator == 0:
                expln = numerator/ (denominator + 1e-6)
            else:
                expln = numerator/ denominator
            #################################            
            
            output_cam.append(expln)
            
        cam = torch.stack(output_cam)

        ######################################################
        ############## Refinement via Deletion Test ################
        total_len = len(text_words)
        granularity = np.linspace(0, 1, 10)
        trunc_words_num = [int(g) for g in np.round(granularity*total_len)]
        trunc_words_num = list(dict.fromkeys(trunc_words_num))
        descending_sorted_idx = torch.argsort(-cam,dim=1).detach().cpu().numpy()

        ######################################################
        # This loop can be parallelized across multiple CAMs
        # (each CAM explanation is independently tested via deletion)
        
        per_cam_trunc_proba_info = []
        for sorted_idx in descending_sorted_idx:
            trunc_proba_list = []

            for num in trunc_words_num[1:]: #exclude 0

                truncated_text_ids_libra, t, _ = self.truncate_words(sorted_idx, text_words, text_ids, num, seg_ids=None)
                trunc_class_libra, trunc_prob_libra = self.predict(truncated_text_ids_libra, index, seg_ids=None)

                diff = original_prob - trunc_prob_libra
                trunc_proba_list.append( diff )
                
            per_cam_trunc_proba_info.append(trunc_proba_list)
        ######################################################
        
        ############################################################
        per_cam_trunc_proba_info = np.array(per_cam_trunc_proba_info)
        mean_per_cam_trunc_proba_info = per_cam_trunc_proba_info.mean(axis=1)

        m_trunc = mean_per_cam_trunc_proba_info.mean()
        std_trunc = mean_per_cam_trunc_proba_info.std()

        threshold = m_trunc + std_trunc
        filterd_idx = (mean_per_cam_trunc_proba_info >= threshold)
        
        if filterd_idx.sum() != 0:
            trunc_choice_cam = cam[filterd_idx]
            trunc_choice_cam = trunc_choice_cam.mean(dim=0)
        else:
            trunc_choice_cam = cam.mean(dim=0)
        ############################################################
        
        ############################################################
        ######## Min-Max scaling ########
        min_v = trunc_choice_cam.min()
        max_v = trunc_choice_cam.max()

        numerator = trunc_choice_cam - min_v
        denominator = max_v - min_v

        if denominator == 0:
            trans_expln = numerator/ (denominator + 1e-6)
        else:
            trans_expln = numerator/ denominator
        ############################################################
        
        return trans_expln.squeeze(0)

    def generate_gradsam(self, input_ids, attention_mask, index=None):
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        
        blocks = self.model.bert.encoder.layer        
        cams = {}
        for blk_id in range(len(blocks)):
            
            cam = blocks[blk_id].attention.self.get_attn().squeeze(0).detach().cpu()
            grad = blocks[blk_id].attention.self.get_attn_gradients().squeeze(0).detach().cpu()
            grad = F.relu(grad)
            
            H = (grad * cam)
            H = H.mean(dim=-1).squeeze(0) # N (Embedding)
            H = H.mean(dim=0) #M (Head)
            cams[blk_id] = H
            
        sam = sum(cams.values())/len(blocks)
        sam = sam.detach().cpu()
        
        trans_expln = sam
        
        #################################
        ######## Min-Max scaling ########
        min_v = trans_expln.min()
        max_v = trans_expln.max()

        numerator = trans_expln - min_v
        denominator = max_v - min_v

        if denominator == 0:
            trans_expln = numerator/ (denominator + 1e-6)
        else:
            trans_expln = numerator/ denominator
        #################################        

        return trans_expln.squeeze(0)

    # compute rollout between attention layers
    def compute_rollout_attention(self, all_layer_matrices, start_layer=0):
        
        # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
        num_tokens = all_layer_matrices[0].shape[1]
        batch_size = all_layer_matrices[0].shape[0]
        eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
        all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
        matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                              for i in range(len(all_layer_matrices))]
        joint_attention = matrices_aug[start_layer]
        for i in range(start_layer+1, len(matrices_aug)):
            joint_attention = matrices_aug[i].bmm(joint_attention)
            
        return joint_attention
    
    def truncate_words(self, sorted_idx, text_words, text_ids, replaced_num, seg_ids=None):
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
    
    def predict(self, text_ids, target, att_mask=None, seg_ids=None):
        out = self.model(text_ids, attention_mask=att_mask, token_type_ids=seg_ids)
        prob = out[0]
        pred_class = torch.argmax(prob, axis=1).cpu().detach().numpy()
        pred_class_prob = F.softmax(prob, dim=1).cpu().detach().numpy()
        return pred_class[0], pred_class_prob[:, target][0]
    ################################################################################################