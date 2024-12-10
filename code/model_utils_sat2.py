import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import AdamW, get_scheduler 
from tqdm import tqdm,trange
import numpy as np
from collections import deque
from copy import deepcopy
import scipy
import ipdb as pdb
import gc

model_classes = {
    'clm': AutoModelForCausalLM,
}

class cm_model(torch.nn.Module):
    def __init__(self, model_name='gpt2-medium', cache_dir="./cache", model_type='clm', load_model=False):
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        model_class = model_classes[model_type]
        self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        if load_model:
            self.model = model_class.from_pretrained(model_name, from_tf=False,
                                                     config=self.config, cache_dir=cache_dir)
            self.model.resize_token_embeddings(len(self.tokenizer))  # Adjust embedding size
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            for p in self.model.parameters():
                p.requires_grad = False  # Disable gradient updates
            self.model.eval()
        else:
            self.model = None
             
    def encode(self, context, past_key_values=None, use_cache=True, enable_grad=False, additional_truncation=0):
        wte = self.model.get_input_embeddings()
        tokens = self.tokenizer(context)
        max_len = self.model.config.n_positions-additional_truncation - 50 #for choice length
        
        if len(tokens['input_ids']) > max_len:
            print ("{} truncating {}->{}".format(context[:80], len(tokens['input_ids']), max_len))
            for k in tokens:
                tokens[k] = tokens[k][-max_len:]

        tokens = {k:torch.tensor(v).unsqueeze(0).to(self.device) for k,v in tokens.items()}
        temp = tokens['input_ids'].detach().clone()
        del tokens['attention_mask']

        inputs_embeds = wte(tokens['input_ids'])
        inputs_embeds.requires_grad = enable_grad
        outputs = self.model(inputs_embeds=inputs_embeds, return_dict=True,
                            past_key_values=past_key_values, use_cache=use_cache)

        # Ensure 'past_key_values' is included in outputs
        if 'past_key_values' not in outputs:
            raise ValueError("The model did not return 'past_key_values'. Ensure caching is enabled.")

        outputs['input_ids'] = temp
        outputs['inputs_embeds'] = inputs_embeds
        return outputs
    
    def compute_nll(self, context_outputs, ch_outputs, enable_grad=True): #conditional
        with torch.set_grad_enabled(enable_grad):
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            
            labels = ch_outputs['input_ids'].contiguous()
            logits = torch.cat([context_outputs['logits'][:, -1:, :], ch_outputs['logits'][:,:-1, :]], dim=1).contiguous()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            prob = F.softmax(logits, dim=2)
                        
        outputs = {
            'prob': prob.detach().cpu().numpy(),
            'labels': labels.view(-1).detach().cpu().numpy(),
            'loss': loss,
            'score': torch.sum(loss),
        }
        return outputs

    def process_context_and_choices(self, context, choices, reduction='sum', out_tokens={}, bag=False, additional_truncation=0):
        enable_grad=False
        scores = []
        for ch_idx in range(len(choices)):
            self.model.zero_grad()
            context_outputs = self.encode(context,
                                  use_cache=True,
                                  enable_grad=enable_grad,
                                  past_key_values=None, additional_truncation=additional_truncation)

            ch_output = self.encode(choices[ch_idx],
                                    past_key_values=context_outputs['past_key_values'],
                                    enable_grad=enable_grad)

            score = self.compute_nll(context_outputs, ch_output, enable_grad=enable_grad)

            scores.append(score)

        return scores

    @torch.inference_mode()
    
    def score(self, example, additional_truncation=0):
        scores = [{} for _ in range(len(example['choices']))]

        #context
        temp_scores = self.process_context_and_choices(example['context'], example['choices'], additional_truncation=additional_truncation)
        for ch_idx in range(len(example['choices'])):
            scores[ch_idx]['res0'] = temp_scores[ch_idx]

        #dummy context
        temp_scores = self.process_context_and_choices(example['dummy_context'], example['choices'])
        for ch_idx in range(len(example['choices'])):
            scores[ch_idx]['res1'] = temp_scores[ch_idx]                 
        
        #save additional results
        for ch_idx in range(len(example['choices'])):
            tcal_prob2 = scores[ch_idx]['res0']['prob'] - scores[ch_idx]['res1']['prob'][:,:1,:]
            tcal_prob2 = scipy.special.softmax(tcal_prob2, axis=2)
            tcal_loss2 = tcal_prob2[:, range(tcal_prob2.shape[1]), scores[ch_idx]['res0']['labels']]
            scores[ch_idx]['res0']['tcal_loss2'] = -np.log(tcal_loss2)

            pc,pn = scores[ch_idx]['res0']['prob'][0,0,:], scores[ch_idx]['res1']['prob'][0,0,:]
            scores[ch_idx]['c_prob0'] = pc.reshape(-1)
            scores[ch_idx]['n_prob0'] = pn.reshape(-1)

            del scores[ch_idx]['res0']['prob']
            del scores[ch_idx]['res1']['prob']
            
        return scores

    def compute_saliency(self, context, choices):
        """
        Computes token-level saliency scores based on the model's likelihood scoring mechanism.
        Args:
            context: The context/prompt text
            choices: List of possible answer choices
        Returns:
            Dictionary containing saliency scores aligned with model's scoring approach
        """
        saliency_scores = []
        
        for choice_idx, choice in enumerate(choices):
            # First pass: Get likelihood scores
            self.model.zero_grad()
            
            # Enable gradient tracking for context
            context_outputs = self.encode(
                context,
                use_cache=True,
                enable_grad=True,
                past_key_values=None
            )
            
            # Get choice outputs with cached context
            choice_outputs = self.encode(
                choice,
                past_key_values=context_outputs['past_key_values'],
                enable_grad=True
            )
            
            # Compute loss (same as in compute_nll)
            score = self.compute_nll(context_outputs, choice_outputs, enable_grad=True)
            
            # Get gradients for both context and choice
            score['loss'].mean().backward()
            
            # Extract saliency scores
            context_saliency = context_outputs['inputs_embeds'].grad.abs().sum(dim=-1)
            choice_saliency = choice_outputs['inputs_embeds'].grad.abs().sum(dim=-1)
            
            # Normalize scores
            context_saliency = context_saliency / context_saliency.max()
            choice_saliency = choice_saliency / choice_saliency.max()
            
            # Map scores back to tokens
            context_tokens = self.tokenizer.convert_ids_to_tokens(
                context_outputs['input_ids'][0]
            )
            choice_tokens = self.tokenizer.convert_ids_to_tokens(
                choice_outputs['input_ids'][0]
            )
            
            saliency_scores.append({
                'choice_idx': choice_idx,
                'context_saliency': list(zip(context_tokens, 
                                          context_saliency[0].cpu().numpy())),
                'choice_saliency': list(zip(choice_tokens, 
                                          choice_saliency[0].cpu().numpy())),
                'score': score['score'].item()
            })
        
        return saliency_scores

    def score_with_saliency(self, example):
        """
        Compute both scores and saliency for an example.
        """
        # Get regular scores (uses existing scoring logic)
        scores = self.score(example)
        
        # Compute saliency aligned with scoring mechanism
        saliency_scores = self.compute_saliency(
            example['context'], 
            example['choices']
        )
        
        return {
            'scores': scores,
            'saliency': saliency_scores
        }

def predict_(scores, mode=None):
    c_scores = np.array([item['res0']['score'] for item in scores])
    n_scores = np.array([item['res1']['score'] for item in scores])
    #cprob = scores[0]['c_prob0']
    #nprob = scores[0]['n_prob0']
    cprob = np.array([item['c_prob0'] for item in scores])
    nprob = np.array([item['n_prob0'] for item in scores])
    if mode == 'alc_unscaled':
        scores = c_scores - n_scores
    elif mode == 'alc_tvd':
        #mult = 1 - 0.5 * np.sum(np.abs(cprob-nprob))
        mult = 1 - 0.5 * np.sum(np.abs(cprob-nprob), axis=1)
        scores = c_scores - mult * n_scores
    elif mode == 'alc_bc':
        #mult = np.sum(np.sqrt(cprob*nprob))
        mult = np.sum(np.sqrt(cprob*nprob), axis=1)
        scores = c_scores - mult * n_scores
    elif mode == 'uncalibrated':
        scores = c_scores
    elif mode == 'length_normalized':
        scores = np.array([np.mean(item['res0']['loss']) for item in scores])
    elif mode == 'answer_only':
        scores = n_scores
    elif mode == 'answer_only_norm':
        scores = np.array([np.mean(item['res1']['loss']) for item in scores])
    elif mode == 'answer_only_worst':
        scores = -n_scores
    elif mode == 'answer_only_worst_norm':
        scores = -np.array([np.mean(item['res1']['loss']) for item in scores])
    elif mode == 'token_calibration':
        scores = np.array([np.sum(item['res0']['tcal_loss2']) for item in scores])
    else:
        raise NotImplementedError("Mode {} not implemented".format(mode))
   
    pred_idx = np.argmin(scores)
    scores_exp = np.exp(-scores)
    conf = scores_exp[pred_idx]/np.sum(scores_exp)

    res = {
            'pred': pred_idx,
            'conf': conf
        }
    return res

def predict(scores, mode='default'):
    out = [predict_(item, mode=mode) for item in scores]
    preds = [item['pred'] for item in out]
    conf = [item['conf'] for item in out]
    info = {}
    info['conf'] = conf
    return preds, info

#for mp
def score_fn(p_rank, model, data, indices, return_dict):
    enable_grad = False
    if p_rank == 0:
        indices = tqdm(indices)
    for idx in indices:
        with torch.set_grad_enabled(enable_grad):
            b_sucesss = False
            for additional_truncation in range(0,1000,50):
                try:
                    gc.collect()
                    outputs = model.score(data[idx], additional_truncation=additional_truncation)
                    b_sucesss = True
                    break
                except Exception as e:
                    excpt = e
                    pass
            if not b_sucesss:
                raise ValueError("Input too long or", excpt)
        return_dict[idx] = outputs
    
