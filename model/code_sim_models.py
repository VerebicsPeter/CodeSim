import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# Hugging Face Transformers
import transformers
from transformers import BatchEncoding
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from tqdm.auto import tqdm

from collections import defaultdict
from typing import Any, Callable


class EvalDataWrapper:
    def __init__(self, data_loader: DataLoader,
                 data_embedder: Callable[[Any, nn.Module, torch.device],Any],
                 aggr_hooks: dict[str, Callable[[Any, defaultdict],None]],
                 metr_hooks: dict[str, Callable],
                 ):
        self.data_loader = data_loader
        self.data_embedder = data_embedder
        assert aggr_hooks.keys() == metr_hooks.keys(), "Hooks must have matching keys!"
        self.metrics = metr_hooks.keys()
        self.aggr_hooks = aggr_hooks
        self.metr_hooks = metr_hooks
        
    
    def embed_data(self, model: nn.Module, device: torch.device) -> list:
        embedded_batches = []
        for data in tqdm(self.data_loader):
            embedded_batch = self.data_embedder(data, model, device)
            embedded_batches.append(embedded_batch)
        # NOTE: return the embedded data of batches
        return embedded_batches

    def calc_mertics(self, embedded_batches):
        aggr_data = {
            metric: defaultdict(list)
            for metric in self.metrics 
        }
        for embedded_batch in embedded_batches:
            for metric in self.metrics:
                self.aggr_hooks[metric](embedded_batch, aggr_data[metric])
        metr_data = {
            metric: 
                self.metr_hooks[metric](**aggr_data[metric])
            for metric in self.metrics
        }
        return metr_data


class CodeSimTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader_wrappers: dict[str, EvalDataWrapper],
        loss_func: Callable,
        loss_hook: Callable,
        optimizer,
        scheduler,
        device: torch.device,
        train_data: Dataset | None = None,
        train_data_embedder: Callable | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        assert len(valid_loader_wrappers) >= 1,\
        "ERROR: Please provide at least one validation loader!"
        self.valid_loader_wrappers = valid_loader_wrappers
        ls = valid_loader_wrappers
        assert sum(len(ls[name].metrics) for name in ls),\
        "ERROR: Please provide at least one validation metric!"
        
        self.loss_func = loss_func
        self.loss_hook = loss_hook
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = GradScaler(self.device)
        
        self.train_data = train_data
        self.train_data_embedder = train_data_embedder    
    
    def train_step(self, iters_to_accumulate: int, print_every: int):
        """Train one epoch."""
        self.model.train()
        
        sum_loss = 0.0
        num_iter = len(self.train_loader)
        if not num_iter:
            raise ValueError("Loader is empty, please check the dataset and dataloader.")
        
        running_loss = 0.0
        for iter, data in enumerate(tqdm(self.train_loader)):
            # Enables autocasting for the forward pass (model + loss)
            with autocast("cuda"):
                loss = self.loss_hook(self, data)
                # Normalize the loss because it is averaged
                loss = loss / iters_to_accumulate

            # Backpropagating the gradients
            # Scales loss. (calls backward() on scaled loss to create scaled gradients)
            self.scaler.scale(loss).backward()

            if (iter + 1) % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                self.scaler.step(self.optimizer)
                # Updates the scale for next iteration.
                self.scaler.update()
                # Adjust the learning rate based on the number of iterations.
                self.scheduler.step()
                # Clear gradients
                self.optimizer.zero_grad()

            running_loss += loss.item()
            # Print training loss information
            if (iter + 1) % print_every == 0:
                print(f"Iteration {iter + 1}/{num_iter} complete. Loss: {running_loss / print_every}")
                sum_loss += running_loss
                running_loss = 0.0
        
        avg_loss = sum_loss / num_iter
        return avg_loss
    
    @torch.no_grad
    def valid_step(self):
        """Evaluate the model on validation data."""
        # Set the model to evaluation mode
        self.model.eval()
        metrics = {}
        
        for loader_name in self.valid_loader_wrappers:
            valid_loader_wrapper = self.valid_loader_wrappers[loader_name]
            if not len(valid_loader_wrapper.data_loader):
                raise ValueError("Loader is empty, please check the dataset and dataloader.")
            embeddings = valid_loader_wrapper.embed_data(self.model, self.device)
            metrics[loader_name] = valid_loader_wrapper.calc_mertics(embeddings)
        
        return metrics

    
    def train(self,
        epochs: int,
        iters_to_accumulate: int = 2,
        print_every: int | None = None,
        cache_every: int | None = None,
    ):
        # Path to save the best model to
        BEST_MODEL_PATH = "best_model.pt"
        LAST_MODEL_PATH = "last_model.pt"
        
        best_metrics = {}
        
        train_losses, valid_losses = [],[]
        metrics_history = []
        
        # Print training loss 5 times per epoch by default
        if print_every is None: 
            print_every = len(self.train_loader) // 5
        # Calculate and cache training set embeddings every epoch by default
        if cache_every is None:
            cache_every = 1
        
        for epoch in range(epochs):
            if epoch % cache_every == 0:
                if self.train_data_embedder is not None: self.train_data_embedder(self.train_data, self.model)
                
            print(f'EPOCH {epoch + 1}/{epochs}')
            train_loss = self.train_step(iters_to_accumulate, print_every)
            valid_loss = np.inf
            valid_metrics = self.valid_step()
            print(f"EPOCH {epoch + 1}/{epochs} complete. AVG train loss: {train_loss}\n"
                  f" METRICS: {valid_metrics}")
            metrics_history.append(valid_metrics)
            
            for loader_name in valid_metrics:
                metrics = valid_metrics[loader_name]
                for metric in metrics:
                    improved = False
                    metric_value = metrics[metric]
                    if (is_loss_mertic := "loss" in metric):
                        valid_loss = metric_value
                    if metric in best_metrics:
                        improved = (
                            metric_value < best_metrics[metric]
                            if is_loss_mertic else
                            metric_value > best_metrics[metric]
                        )
                    else:
                        best_metrics[metric] = metric_value
                        torch.save(self.model.state_dict(), BEST_MODEL_PATH)
                    if improved:
                        best_metrics[metric] = metric_value
                        print(f"Best metrics improved, saving state.")
                        torch.save(self.model.state_dict(), BEST_MODEL_PATH)

            torch.cuda.empty_cache()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
        losses_history = {
            "train": train_losses,
            "valid": valid_losses,
        }
        
        torch.save(metrics_history, "metrics.pickle")
        torch.save(losses_history, "losses.pickle")
        torch.save(self.model.state_dict(), LAST_MODEL_PATH)
        
        return train_losses, valid_losses


"""
NOTE: Pooling strategy returns a pooled output based on the output of a transformer, this could be the:
 - `pooler_output` which is the embedding of the CLS token in BERT models,
 - a mean pooled version of the last hidden states
 - a max  pooled version of the last hidden states
 - an attention pooled version of the last hidden states
"""
# TODO: RNN (GRU) based pooling strategy
PoolingStrategy = Callable[[BaseModelOutputWithPooling | torch.Tensor, torch.Tensor], torch.Tensor]


def cls_pooling_strat(output: BaseModelOutputWithPooling, *_):
    return output.pooler_output


def max_pooling_strat(output: BaseModelOutputWithPooling, mask: torch.Tensor):
    pooled_output = output.last_hidden_state * mask.unsqueeze(-1)  # mask out irrelevant embeddings
    return pooled_output.max(dim=1).values


def mean_pooling_strat(output: BaseModelOutputWithPooling, mask: torch.Tensor):
    pooled_output = output.last_hidden_state * mask.unsqueeze(-1)  # mask out irrelevant embeddings
    return pooled_output.mean(dim=1)


def qwen3_pooling_strat(output: BaseModelOutput, mask: torch.Tensor):
    left_padding = (mask[:, -1].sum() == mask.shape[0])
    last_hidden_state = output.last_hidden_state
    if left_padding:
        return last_hidden_state[:, -1]
    else:
        sequence_lengths = mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


class AttentionPooler(nn.Module):
    def __init__(self, bert_dim, attn_dim):
        super().__init__()
        # TODO: maybe initialize linear layers with uniform weights
        self.attention = nn.Sequential(
            nn.Linear(bert_dim,
                      attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
    
    def forward(self, output: BaseModelOutputWithPooling, mask: torch.Tensor):
        # get attention scores using learnable weights
        attn_scores = self.attention(output.last_hidden_state).squeeze(-1)
        # mask attention scores with '-inf', softmax turns them to 0...
        attn_scores = attn_scores.masked_fill(mask==0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        # Return the weighted representation
        return (output.last_hidden_state * attn_weights).sum(dim=1)


def freeze_model(model: nn.Module):
    for param in model.parameters(): param.requires_grad = False


def put_batch_encoding_to_device(encoding: BatchEncoding, device):
    for k, v in encoding.items(): encoding[k] = v.to(device)


class CodeSimLinearClassifierCross(nn.Module):
    def __init__(
        self,
        enc_model: transformers.PreTrainedModel,
        freeze_enc_model=False,
        dropout_rate=0.2,
        pooling_strat: PoolingStrategy = cls_pooling_strat
    ):
        super().__init__()
        if freeze_enc_model: freeze_model(enc_model)
        self.enc_model = enc_model
        self.pooling_strat = pooling_strat
        self.drop = nn.Dropout(dropout_rate)
        self.cls = nn.Linear(self.enc_model.config.hidden_size, 1)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        mask = inputs['attention_mask']
        output: BaseModelOutputWithPooling = self.enc_model(**inputs)
        pooled_output = self.pooling_strat(output, mask)
        logits = self.cls(self.drop(pooled_output))
        return logits


class CodeSimLinearClassifierSBert(nn.Module):
    def __init__(
        self, 
        enc_model: transformers.PreTrainedModel,
        freeze_enc_model=False,
        dropout_rate=0.2,
        pooling_strat: PoolingStrategy = cls_pooling_strat,
    ):
        super().__init__()
        if freeze_enc_model: freeze_model(enc_model)
        self.enc_model = enc_model
        self.pooling_strat = pooling_strat
        self.drop = nn.Dropout(dropout_rate)
        self.cls = nn.Linear(3 * enc_model.config.hidden_size, 1)  # weights for concatenated [ u, v, |u - v| ]
    
    def forward(self, enc_u: BatchEncoding, enc_v: BatchEncoding) -> torch.Tensor:
        mask_u = enc_u['attention_mask']
        mask_v = enc_v['attention_mask']
        # Pass through BERT
        u = self.enc_model(**enc_u)
        v = self.enc_model(**enc_v)
        # Pool the BERT output
        pooled_u = self.pooling_strat(u, mask_u)
        pooled_v = self.pooling_strat(v, mask_v)
        # Construct the feature vector [ u, v, |u - v| ]
        h = torch.cat([pooled_u, pooled_v, torch.abs(pooled_u - pooled_v)], dim=1)
        # Classification layer
        logits = self.cls(self.drop(h))
        return logits


class CodeSimContrastiveEncoder(nn.Module):
    def __init__(
        self,
        enc_model: transformers.PreTrainedModel,
        freeze_enc_model=False,
        dropout_rate=0.2,
        pooling_strat: PoolingStrategy = cls_pooling_strat
    ):
        super().__init__()
        if freeze_enc_model: freeze_model(enc_model)
        self.enc_model = enc_model
        self.enc_tokenizer = None
        self.pooling_strat = pooling_strat
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        mask = inputs['attention_mask']
        output: BaseModelOutputWithPooling = self.enc_model(**inputs)
        pooled_output = self.drop(self.pooling_strat(output, mask))
        return pooled_output


# TODO: Make temp a learnable parameter
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, local_loss_func=None, w1=1.0, w2=1.0, temp=0.07, device=None):
        """
        Args:
            local_loss_func: function (A, POS, NEG) -> scalar tensor
            w1 (float): weight for global cross-entropy loss
            w2 (float): weight for local loss
            temp (float): temperature scaling for similarity
            device (torch.device or None): device for labels
        """
        super().__init__()
        self.local_loss_func = local_loss_func
        self.w1 = w1
        self.w2 = w2
        self.temp = temp
        self.device = device

    def forward(self, anchor, pos, neg):
        """
        Given N DIFFERENT classes with
        
        `a,p,n` examples for each class (`a,p` positive and `n` negative)
        
        Computes the loss on the embeddings of examples as follows:
        1. Concatenate the embeddings of `p`, `n` examples (`C`).
        2. Compute the cosine similarities between the anchors and `C`.
        3. Scale the cosine similarities by `temp` to obtain logits.
        4. Compute the cross-entropy loss between the logits and labels.
        
        NOTE:  
        This **requires** the batched tuplets to be from **different problems**.  
        If the some tuplets are from the same problem then labels are incorrect.
        """
        # L2 normalize embeddings for cosine similarity
        anchor = F.normalize(anchor, dim=1)
        pos = F.normalize(pos, dim=1)
        neg = F.normalize(neg, dim=1)

        N = anchor.size(0)

        # Concatenate positives and negatives as candidates
        C = torch.cat([pos, neg], dim=0)     # shape: (2N, d)

        # Similarity logits (cosine similarity / temperature)
        logits = (anchor @ C.T) / self.temp  # shape: (N, 2N)

        # Ground-truth labels: each A should match its POS
        labels = torch.arange(N, device=anchor.device if self.device is None else self.device)

        # Global cross-entropy loss
        loss_1 = F.cross_entropy(logits, labels)

        # Local loss from provided function
        loss_2 = self.local_loss_func(anchor, pos, neg) if self.local_loss_func else 0

        return self.w1 * loss_1 + self.w2 * loss_2


def compute_loss_logit_Cross(trainer: CodeSimTrainer, batched_data):
    """Loss strategy for fine tuning BERT."""
    encoding, labels = batched_data
    # Converting to cuda tensors if needed
    put_batch_encoding_to_device(encoding, trainer.device)
    # also convert labels...
    labels = labels.to(trainer.device)
    # Obtaining the logits from the model
    logits = trainer.model(encoding)
    # Computing loss
    loss = trainer.loss_func(logits.squeeze(-1), labels.float())
    return loss


def compute_loss_logit_SBert(trainer: CodeSimTrainer, batched_data):
    """Loss strategy for finetuning BERT."""
    encoding_u, encoding_v, labels = batched_data
    # Converting to cuda tensors if needed
    put_batch_encoding_to_device(encoding_u, trainer.device)
    put_batch_encoding_to_device(encoding_v, trainer.device)
    # also convert labels...
    labels = labels.to(trainer.device)
    # Obtaining the logits from the model
    logits = trainer.model(encoding_u, encoding_v)
    # Computing loss
    loss = trainer.loss_func(logits.squeeze(-1), labels.float())
    return loss


def compute_loss_triplet(trainer: CodeSimTrainer, batched_data):
    """Loss strategy for finetuning BERT."""
    encs_a, encs_p, encs_n = batched_data
    N = encs_a["input_ids"].shape[0]  # batch size
    inputs = { key: torch.cat([encs_a[key], encs_p[key], encs_n[key]]) for key in encs_a }
    # Converting to cuda tensors if needed
    put_batch_encoding_to_device(inputs, trainer.device)
    embs = trainer.model(inputs)
    embs_a, embs_p, embs_n = embs.split(N)
    return trainer.loss_func(embs_a, embs_p, embs_n)


def compute_loss_tuplet(trainer: CodeSimTrainer, batched_data):
    """
    Loss strategy for finetuning BERT.
    """
    
    enc_a, enc_p, encs_n = batched_data
    N = enc_a["input_ids"].shape[0]  # batch size
    inputs = { key: torch.cat([enc_a[key], enc_p[key], encs_n[key]]) for key in enc_a }
    
    put_batch_encoding_to_device(inputs, trainer.device)
    
    embs = trainer.model(inputs)
    
    A, POS, NEG = embs.split((N, N, embs.size(0)-2*N))
    return trainer.loss_func(A, POS, NEG)


def compute_loss_combined(trainer: CodeSimTrainer, batched_data):  
    enc_a, enc_p, encs_n = batched_data
    N = enc_a["input_ids"].shape[0]  # batch size
    inputs = { key: torch.cat([enc_a[key], enc_p[key], encs_n[key]]) for key in enc_a }
    
    put_batch_encoding_to_device(inputs, trainer.device)
    
    embs = trainer.model(inputs)
    
    A, POS, NEG = embs.split(N)
    return trainer.loss_func(A, POS, NEG)


# TODO: Combined loss function for contrastive and classification objectives


def pass_data_classifier(data, model: CodeSimLinearClassifierCross | CodeSimLinearClassifierSBert, device):
    if not isinstance(model, (CodeSimLinearClassifierCross, CodeSimLinearClassifierSBert)):
        raise ValueError(f"Invalid model type {type(model)}")
    if isinstance(model, CodeSimLinearClassifierCross):
        encs, labels = data
        put_batch_encoding_to_device(encs, device)
        logits = model.forward(encs)
    if isinstance(model, CodeSimLinearClassifierSBert):
        encs_u, encs_v, labels = data
        put_batch_encoding_to_device(encs_u, device)
        put_batch_encoding_to_device(encs_v, device)
        logits = model.forward(encs_u, encs_v)
    return logits, labels

def aggr_data_classifier(data, aggr_data: defaultdict[str, list]):
    logits, labels = data
    preds_ = torch.sigmoid(logits.squeeze(-1))
    aggr_data["y_true"].extend(labels.cpu().tolist())
    aggr_data["y_pred"].extend(preds_.cpu().tolist())


def embd_data_contrastive_cls(data, model: CodeSimContrastiveEncoder, device: torch.device):
    encs_a, encs_p, encs_n = data
    batch_size = encs_a["input_ids"].shape[0]
    # Combine the inputs into single encoding
    inputs = {key: torch.cat([encs_a[key], encs_p[key], encs_n[key]]) for key in encs_a}
    put_batch_encoding_to_device(inputs, device)
    outputs = model.forward(inputs)
    embs_a, embs_p, embs_n = outputs.split(batch_size)
    return embs_a, embs_p, embs_n

def aggr_data_contrastive_cls(data, aggr_data: defaultdict[str, list]):
    embs_a, embs_p, embs_n = data
    # Calculate the pairwise cosine similarities
    dst_p = F.cosine_similarity(embs_a, embs_p, dim=1)
    dst_n = F.cosine_similarity(embs_a, embs_n, dim=1)
    preds_p = dst_p.cpu().tolist()
    preds_n = dst_n.cpu().tolist()
    # Store predictions and labels
    aggr_data['y_true'].extend([1] * len(preds_p))
    aggr_data['y_true'].extend([0] * len(preds_n))
    aggr_data['y_pred'].extend(preds_p)
    aggr_data['y_pred'].extend(preds_n)


def embd_data_contrastive_cls_pair(data, model: CodeSimContrastiveEncoder, device: torch.device):
    encs_1 = {
        "input_ids": data["program_1_input_ids"],
        "attention_mask": data["program_1_attention_mask"],
    }
    encs_2 = {
        "input_ids": data["program_2_input_ids"],
        "attention_mask": data["program_2_attention_mask"],
    }
    lbls = data["num_truth_label"]
    keys = encs_1.keys() & encs_2.keys()

    batch_size = encs_1["input_ids"].shape[0]
    inputs = {key: torch.cat([encs_1[key], encs_2[key]]) for key in keys}
    put_batch_encoding_to_device(inputs, device)
    outputs = model.forward(inputs)
    embs_1, embs_2 = outputs.split(batch_size)
    return embs_1, embs_2, lbls

def aggr_data_contrastive_cls_pair(data, aggr_data: defaultdict[str, list]):
    embs_1, embs_2, lbls = data
    sims = F.cosine_similarity(embs_1, embs_2, dim=1)
    trues = lbls.cpu().tolist()
    preds = sims.cpu().tolist()
    aggr_data["y_true"].extend(trues)
    aggr_data["y_pred"].extend(preds)


def embd_data_contrastive_map(data, model: CodeSimContrastiveEncoder, device: torch.device):
    encs, lbls = data
    put_batch_encoding_to_device(encs, device)
    embs = model.forward(encs)
    return embs, lbls

def aggr_data_contrastive_map(data, aggr_data: defaultdict[str, list]):
    embs, lbls = data
    aggr_data["all_embs"].append(embs.detach().cpu())
    aggr_data["all_lbls"].append(lbls.detach().cpu())
