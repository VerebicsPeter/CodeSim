import numpy as np
import copy

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
#from pytorch_metric_learning import losses

# Hugging Face Transformers
import transformers
from transformers import AutoTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from tqdm.auto import tqdm

from collections import defaultdict
from typing import Callable, Protocol, Tuple

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


def unwrap_model(model: nn.Module) -> nn.Module:
    """ Unwraps the model from the trainer if it is wrapped in a trainer."""
    if isinstance(model, nn.DataParallel):
        return model.module
    else:
        return model


def put_batch_encoding_to_device(encoding: BatchEncoding, device):
    for k, v in encoding.items(): encoding[k] = v.to(device)


def get_tokenizer(model: transformers.PreTrainedModel) -> transformers.PreTrainedTokenizerBase:
    if (not model.name_or_path): raise ValueError("Model's name or path is not known.")
    return AutoTokenizer.from_pretrained(model.name_or_path)


class Trainer(Protocol):
    # Returns train and validation losses in a tuple
    def train(self, epochs: int, **kwargs) -> Tuple: ...


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


class CodeSimilarityTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        loaders: Tuple[DataLoader, DataLoader],
        loss_func: Callable,
        loss_hook: Callable,
        optimizer,
        scheduler,
        device: torch.device,
        loss_checkpointing: bool = True,  # Whether to save the best model based on validation loss
        target_metrics: list[str] | None = None,
        aggr_hook: Callable | None = None,
        compute_metrics: Callable[[],dict[str,]] | None = None,
    ):
        self.model = model
        assert len(loaders) == 2, "Please provide the training and validation loaders!"
        self.train_loader = loaders[0]
        self.valid_loader = loaders[1]
        
        self.loss_func = loss_func
        self.loss_hook = loss_hook
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = GradScaler(self.device)
        
        self.loss_checkpointing = loss_checkpointing
        self.target_metrics = target_metrics
        self.aggr_hook = aggr_hook
        self.compute_metrics = compute_metrics
    
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
        
        aggr_data = defaultdict(list)
        
        sum_loss = 0.0
        num_iter = len(self.valid_loader)
        if not num_iter:
            raise ValueError("Loader is empty, please check the dataset and dataloader.")
        
        if self.target_metrics:
            assert self.aggr_hook is not None,\
            f"Metrics calculation requires a data aggregator hook function."
        
        for data in tqdm(self.valid_loader):
            if self.loss_checkpointing:
                loss = self.loss_hook(self, data)
                sum_loss += loss.item()
            if self.target_metrics:
                _ = self.aggr_hook(unwrap_model(self.model), data, aggr_data)
        
        if self.target_metrics:
            metrics = self.compute_metrics(**aggr_data)
            assert set(metrics.keys()) == set(self.target_metrics),\
            f"Metrics computed do not match the target metrics: {self.target_metrics}"
        else:
            metrics = {}
            
        if self.loss_checkpointing:
            avg_loss = sum_loss / num_iter
        else:
            avg_loss = np.inf
        
        return avg_loss, metrics

    
    def train(self, epochs: int, iters_to_accumulate: int = 2):
        # Path to save the best model to
        BEST_MODEL_PATH_EVAL = "best_model_by_eval.pt"
        BEST_MODEL_PATH_LOSS = "best_model_by_loss.pt"
        LAST_MODEL_PATH = "last_model.pt"
        
        best_loss = np.inf
        best_metrics = {metric: -np.inf for metric in self.target_metrics}
        
        train_losses, valid_losses = [],[]
        # Print training loss 5 times per epoch
        print_every = len(self.train_loader) // 5
        
        for epoch in range(epochs):
            print(f'EPOCH {epoch + 1}/{epochs}')
            train_loss = self.train_step(iters_to_accumulate, print_every)
            valid_loss, valid_metrics = self.valid_step()
            print(f"EPOCH {epoch + 1}/{epochs} complete.\n"
                  f" AVG train loss: {train_loss}\n"
                  f" AVG valid loss: {valid_loss}\n"
                  f" METRICS: {valid_metrics}")
            
            if valid_loss < best_loss:
                print(f"Best validation loss improved from {best_loss} to {valid_loss}.")
                best_loss = valid_loss
                torch.save(self.model.state_dict(), BEST_MODEL_PATH_LOSS)
            
            improved_metrics = any(valid_metrics.get(m, -np.inf) > best_metrics[m]
                                   for m in self.target_metrics)
            if improved_metrics:
                print(f"Best metrics improved.")
                for m in self.target_metrics:
                    best_metrics[m] = max(valid_metrics.get(m, -np.inf), best_metrics[m])
                torch.save(self.model.state_dict(), BEST_MODEL_PATH_EVAL)
            
            torch.cuda.empty_cache()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        
        torch.save(self.model.state_dict(), LAST_MODEL_PATH)
        
        return train_losses, valid_losses


def compute_loss_logit_Cross(trainer: CodeSimilarityTrainer, batched_data):
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


def compute_loss_logit_SBert(trainer: CodeSimilarityTrainer, batched_data):
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


def compute_loss_triplet(trainer: CodeSimilarityTrainer, batched_data):
    """Loss strategy for finetuning BERT."""
    encs_a, encs_p, encs_n = batched_data
    N = encs_a["input_ids"].shape[0]  # batch size
    inputs = { key: torch.cat([encs_a[key], encs_p[key], encs_n[key]]) for key in encs_a }
    # Converting to cuda tensors if needed
    put_batch_encoding_to_device(inputs, trainer.device)
    embs = trainer.model(inputs)
    embs_a, embs_p, embs_n = embs.split(N)
    return trainer.loss_func(embs_a, embs_p, embs_n)


# TODO: Make temp a learnable parameter!
def compute_loss_tuplet(trainer: CodeSimilarityTrainer, batched_data, temp=0.05):
    """
    Loss strategy for finetuning BERT.

    Given N DIFFERENT problems in CodeNet with
    
    `a,p,n` solutions for each problem (`a,p` passing and `n` failing)
    
    Computes the loss on the embeddings of solutions as follows:
    1. Concatenate the embeddings of `p`, `n` solutions (`Q`).
    2. Compute the cosine similarities between the anchors and `Q`.
    3. Scale the cosine similarities by `temp` to obtain logits.
    4. Compute the cross-entropy loss between the logits and labels.
    
    NOTE:  
    This **requires** the batched tuplets to be from **different problems**.  
    If the some tuplets are from the same problem then labels are incorrect.
    """
    
    enc_a, enc_p, encs_n = batched_data
    N = enc_a["input_ids"].shape[0]  # batch size
    inputs = { key: torch.cat([enc_a[key], enc_p[key], encs_n[key]]) for key in enc_a }
    
    put_batch_encoding_to_device(inputs, trainer.device)
    
    embs = trainer.model(inputs)
    embs = F.normalize(embs, dim=1)  # normalize to align with cosine similarity
    
    A, POS, NEG = embs.split((N, N, embs.size(0)-2*N))
    Q = torch.cat([POS, NEG], dim=0)
    
    logits = A @ Q.T / temp
    labels = torch.arange(N, device=trainer.device)
    
    return F.cross_entropy(logits, labels)


# TODO: Make temp a learnable parameter!
def compute_loss_combined(
    trainer: CodeSimilarityTrainer, batched_data,
    temp=0.05,
    w_1=1.0,
    w_2=1.0,
):  
    enc_a, enc_p, encs_n = batched_data
    N = enc_a["input_ids"].shape[0]  # batch size
    inputs = { key: torch.cat([enc_a[key], enc_p[key], encs_n[key]]) for key in enc_a }
    
    put_batch_encoding_to_device(inputs, trainer.device)
    
    embs = trainer.model(inputs)
    embs = F.normalize(embs, dim=1)  # normalize to align with cosine similarity
    
    A, POS, NEG = embs.split(N)
    Q = torch.cat([POS, NEG], dim=0)
    
    logits = A @ Q.T / temp
    labels = torch.arange(N, device=trainer.device)
    
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = trainer.loss_func(A, POS, NEG)  # Local loss
    return w_1 * loss_1 + w_2 * loss_2


# TODO: Combined loss function for contrastive and classification objectives


def aggr_data_classifier(model: CodeSimLinearClassifierCross | CodeSimLinearClassifierSBert, data, aggr_data: defaultdict[str, list]):
    if isinstance(model, CodeSimLinearClassifierCross):
        encs, labels = data
        put_batch_encoding_to_device(encs, model.enc_model.device)
        logits = model.forward(encs)
    else:
        encs_u, encs_v, labels = data
        put_batch_encoding_to_device(encs_u, model.enc_model.device)
        put_batch_encoding_to_device(encs_v, model.enc_model.device)
        logits = model.forward(encs_u, encs_v)
    preds = torch.sigmoid(logits.squeeze(-1))
    # Store predictions and labels
    aggr_data["y_true"].extend(labels.cpu().tolist())
    aggr_data["y_pred"].extend(preds .cpu().tolist())


def aggr_data_contrastive_cls(model: CodeSimContrastiveEncoder, data, aggr_data: defaultdict[str, list]):
    encs_a, encs_p, encs_n = data
    batch_size = encs_a["input_ids"].shape[0]
    # Combine the inputs into single encoding
    inputs = {key: torch.cat([encs_a[key], encs_p[key], encs_n[key]]) for key in encs_a}
    put_batch_encoding_to_device(inputs, model.enc_model.device)
    outputs = model.forward(inputs)
    embs_a, embs_p, embs_n = outputs.split(batch_size)
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


def aggr_data_contrastive_map(model: CodeSimContrastiveEncoder, data, aggr_data: defaultdict[str, list]):
    encs, lbls = data
    put_batch_encoding_to_device(encs, model.enc_model.device)
    embs = model.forward(encs)
    aggr_data["all_embs"].append(embs.detach().cpu())
    aggr_data["all_lbls"].append(lbls.detach().cpu())
