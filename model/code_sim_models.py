import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses

# Hugging Face Transformers (CodeBERT etc.)
import transformers
from transformers import AutoTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling

# Libraries for logging
from tqdm.auto import tqdm

from typing import Callable, Iterable, Protocol, Tuple

# NOTE: Pooling strategy is supposed to return a pooled output based on the output of a transformer, this could be the:
# - `pooler_output` which is the embedding of the CLS token in BERT models,
# - a mean pooled version of the last hidden states
# - a max  pooled version of the last hidden states
PoolingStrategy = Callable[[BaseModelOutputWithPooling, torch.Tensor], torch.Tensor]

# TODO: maybe make mask tensor optional
def cls_pooling_strat(output: BaseModelOutputWithPooling, mask: torch.Tensor):
    return output.pooler_output

def max_pooling_strat(output: BaseModelOutputWithPooling, mask: torch.Tensor):
    pooled_output = output.last_hidden_state * mask  # mask out irrelevant embeddings
    return pooled_output.max(dim=1).values

def mean_pooling_strat(output: BaseModelOutputWithPooling, mask: torch.Tensor):
    pooled_output = output.last_hidden_state * mask  # mask out irrelevant embeddings
    return pooled_output.mean(dim=1)


def freeze_model(model: nn.Module):
    for param in model.parameters(): param.requires_grad = False


def put_batch_encoding_to_device(encoding: BatchEncoding, device):
    for k, v in encoding.items(): encoding[k] = v.to(device)

def get_tokenizer(model: transformers.PreTrainedModel) -> transformers.PreTrainedTokenizerBase:
    if (not model.name_or_path): raise ValueError("Model's name or path is not known.")
    return AutoTokenizer.from_pretrained(model.name_or_path)


class Trainer(Protocol):
    # Returns train and validation losses in a tuple
    def train(self, epochs: int, **kwargs) -> Tuple:
        ...


class SimilarityClassifier(Protocol):
    def predict(self, code_a: str | Iterable[str], code_b: str | Iterable[str], **kwargs):
        ...


class CodeSimLinearCLS(nn.Module, SimilarityClassifier):
    def __init__(
        self,
        bert: transformers.BertModel,  # BERT based model instance
        freeze_bert=False,
        dropout_rate=0.2,
        pooling_strat: PoolingStrategy = cls_pooling_strat
    ):
        super().__init__()
        if freeze_bert: freeze_model(bert)
        self.bert = bert
        self.bert_tokenizer = None
        self.pooling_strat = pooling_strat
        self.drop = nn.Dropout(dropout_rate)
        self.cls = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        # Pass through BERT
        output: BaseModelOutputWithPooling = self.bert(**inputs)
        # Pool output
        pooled_output = self.pooling_strat(output, None)
        # Classification layer
        logits = self.cls(self.drop(pooled_output))
        return logits
    
    def predict(self, code_a: str|Iterable[str], code_b: str|Iterable[str], threshold=0.5):
        if self.bert_tokenizer is None: self.bert_tokenizer = get_tokenizer(self.bert)
        
        inputs = self.bert_tokenizer(
            code_a, code_b,
            padding='max_length',  # Pad to max_length
            truncation=True,       # Truncate to max_length
            return_tensors='pt'    # Return torch.Tensor objects
        )
        # Put tensors to current device
        put_batch_encoding_to_device(inputs, self.bert.device)
        
        logits = self.forward(inputs)
        
        pred = (torch.sigmoid(logits.squeeze(-1)) > threshold).int()
        return pred


class CodeSimSBertTripletCLS(nn.Module, SimilarityClassifier):
    def __init__(
        self,
        bert: transformers.BertModel,  # BERT based model instance
        freeze_bert=False,
        dropout_rate=0.2,
        pooling_strat: PoolingStrategy = cls_pooling_strat
    ):
        super().__init__()
        if freeze_bert: freeze_model(bert)
        self.bert = bert
        self.bert_tokenizer = None
        self.pooling_strat = pooling_strat
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        mask = inputs['attention_mask'].unsqueeze(-1)  # Unsqueeze for broadcasting
        # Pass through BERT
        output: BaseModelOutputWithPooling = self.bert(**inputs)
        # Pool the transformer output
        pooled_output = self.pooling_strat(output, mask)
        pooled_output = self.drop(pooled_output)
        return pooled_output
    
    def predict(self, code_a: str|Iterable[str], code_b: str|Iterable[str], threshold=0.5):
        if self.bert_tokenizer is None: self.bert_tokenizer = get_tokenizer(self.bert)
        
        if isinstance(code_a, str) and isinstance(code_b, str):
            codes = [code_a, code_b]
        else:
            assert len(code_a) == len(code_b), "Number of paired sequences MUST match!"
            codes = [*code_a, *code_b]
        
        inputs = self.bert_tokenizer(
            codes,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Put tensors to current device
        put_batch_encoding_to_device(inputs, self.bert.device)
        
        outputs = self.forward(inputs)
        
        # Calculate the pairwise cosine similarities
        mid = len(codes)//2
        sim = F.cosine_similarity(outputs[:mid,:],outputs[mid:,:])
        # Scale and shift cosine similarity to [0,1]
        sim = (sim + 1) / 2
        
        # TODO:
        # Add a TRAINABLE sigmoid here with the cosine similarity as an input feature,
        # scaling and shifting would have to be removed...
        pred = (sim > threshold).int()
        return pred


class CodeSimSBertLinearCLS(nn.Module):
    def __init__(
        self, 
        bert: transformers.BertModel,
        freeze_bert=False,
        dropout_rate=0.2,
        pooling_strat: PoolingStrategy = cls_pooling_strat,
    ):
        super().__init__()
        if freeze_bert: freeze_model(bert)
        self.bert = bert
        self.bert_tokenizer = None
        self.pooling_strat = pooling_strat
        self.drop = nn.Dropout(dropout_rate)
        # Weights for concatenated [ u, v, |u - v| ]
        self.cls = nn.Linear(3 * bert.config.hidden_size, 1)
    
    def forward(self, enc_u: BatchEncoding, enc_v: BatchEncoding):
        mask_u = enc_u['attention_mask'].unsqueeze(-1)  # Unsqueeze for broadcasting
        mask_v = enc_v['attention_mask'].unsqueeze(-1)  # Unsqueeze for broadcasting
        # Pass through BERT
        u = self.bert(**enc_u)
        v = self.bert(**enc_v)
        # Pool the transformer output
        pooled_u = self.pooling_strat(u, mask_u)
        pooled_v = self.pooling_strat(v, mask_v)
        # Construct the feature vector [ u, v, |u - v| ]
        h = torch.cat([pooled_u, pooled_v, torch.abs(pooled_u - pooled_v)], dim=1)
        # Classification layer
        logits = self.cls(self.drop(h))
        return logits

    def predict(self, code_a: str|Iterable[str], code_b: str|Iterable[str], threshold=0.5):
        if self.bert_tokenizer is None: self.bert_tokenizer = get_tokenizer(self.bert)
        
        params = {
            "padding":'max_length',  # Pad to max_length
            "truncation":True,       # Truncate to max_length
            "return_tensors":'pt'    # Return torch.Tensor objects
        }
        
        enc_u = self.bert_tokenizer(code_a, **params)
        enc_v = self.bert_tokenizer(code_b, **params)
        
        # Put tensors to current device
        put_batch_encoding_to_device(enc_u, self.bert.device)
        put_batch_encoding_to_device(enc_v, self.bert.device)
        
        logits = self.forward(enc_u, enc_v)
        
        pred = (torch.sigmoid(logits.squeeze(-1)) > threshold).int()
        return pred


class CodeSimContrastiveCLS(nn.Module, SimilarityClassifier):
    def __init__(
        self,
        # BERT based model instance
        bert: transformers.BertModel,
        freeze_bert=False,
        dropout_rate=0.2,
        # MLP hidden sizes and out features
        mlp_sizes=(768, 512, 256),
        mlp_feat_out=256,
    ):
        super().__init__()
        
        if freeze_bert: freeze_model(bert)
        self.bert = bert
        self.bert_tokenizer = None
        
        # Non linearity
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

        # MLP 'projection head'
        assert len(mlp_sizes) == 3, 'MLP must have 3 hidden sizes.'
        # NOTE: Input size depends on the BERT model
        mlp_feat_in = bert.config.hidden_size
        mlp_layers = []
        mlp_layers.append(nn.Linear(mlp_feat_in, mlp_sizes[0]))
        mlp_layers.extend([self.relu, self.drop])
        mlp_layers.append(nn.Linear(mlp_sizes[0], mlp_sizes[1]))
        mlp_layers.extend([self.relu, self.drop])
        mlp_layers.append(nn.Linear(mlp_sizes[1], mlp_sizes[2]))
        mlp_layers.extend([self.relu, self.drop])
        mlp_layers.append(nn.Linear(mlp_sizes[2], mlp_feat_out))
        self.proj = nn.Sequential(*mlp_layers)

    def forward(self, inputs: BatchEncoding, use_proj=True) -> torch.Tensor:
        # Pass through BERT
        output = self.bert(**inputs)
        # Use the pooler output of BERT
        output = output.pooler_output
        # Pass through projection head layers if needed
        # This is the default behavior, used for training,
        # because loss is calculated on the projection head's dimension.
        if use_proj:
            return self.proj(output)
        else:
        # This should be used for downstream tasks, like predicting equivalence.
            return output

    def predict(self, code_a: str|Iterable[str], code_b: str|Iterable[str], threshold=0.5):
        if self.bert_tokenizer is None: self.bert_tokenizer = get_tokenizer(self.bert)
        
        if isinstance(code_a, str) and isinstance(code_b, str):
            codes = [code_a, code_b]
        else:
            assert len(code_a) == len(code_b), "Number of paired sequences MUST match!"
            codes = [*code_a, *code_b]
        
        inputs = self.bert_tokenizer(
            codes,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Put tensors to current device
        put_batch_encoding_to_device(inputs, self.bert.device)
        
        outputs = self.forward(inputs, use_proj=False)
        
        # Calculate the pairwise cosine similarities
        mid = len(codes)//2
        sim = F.cosine_similarity(outputs[:mid,:],outputs[mid:,:])
        # Scale and shift cosine similarity to [0,1]
        sim = (sim + 1) / 2
        
        # TODO:
        # Add a TRAINABLE sigmoid here with the cosine similarity as an input feature,
        # scaling and shifting would have to be removed...
        pred = (sim > threshold).int()
        return pred


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
    ):
        self.model = model
        assert len(loaders) == 2, "Please provide train and validation loaders!"
        self.train_loader = loaders[0]
        self.valid_loader = loaders[1]
        self.loss_func = loss_func
        self.loss_hook = loss_hook
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = GradScaler('cuda')
    
    def train_epoch(self, iters_to_accumulate: int, print_every: int):
        sum_loss = 0.0
        
        running_loss = 0.0
        self.model.train()

        num_iter = len(self.train_loader)
        for iter, data in enumerate(tqdm(self.train_loader)):
            # Enables autocasting for the forward pass (model + loss)
            with autocast('cuda'):
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
        
        # Return the average evaluation loss
        avg_loss = sum_loss / num_iter
        return avg_loss
    
    def validate(self):
        """Evaluate the model on validation data."""
        # Set the model to evaluation mode
        self.model.eval()
        sum_loss, num_iter = 0,0
        with torch.no_grad():
            for data in tqdm(self.valid_loader):
                loss = self.loss_hook(self, data)
                sum_loss += loss.item()
                num_iter += 1
        # Return the average evaluation loss
        avg_loss = sum_loss / num_iter
        return avg_loss
    
    def train(self, epochs: int, iters_to_accumulate: int = 2):
        # Path to save the best model to
        BEST_MODEL_PATH = "best_model.pt"
        
        best_loss = np.Inf
        train_losses, valid_losses = [],[]
        # Print training loss 5 times per epoch
        print_every = len(self.train_loader) // 5
        
        for epoch in range(epochs):
            print(f'EPOCH {epoch + 1}/{epochs}')
            train_loss = self.train_epoch(iters_to_accumulate, print_every)
            valid_loss = self.validate()
            print(f"EPOCH {epoch + 1}/{epochs} complete. AVG loss: {train_loss}, AVG validation loss: {valid_loss}")
            
            if valid_loss < best_loss:
                print(f"Best validation loss improved from {best_loss} to {valid_loss}.")
                best_loss = valid_loss
                torch.save(self.model.state_dict(), BEST_MODEL_PATH)
            
            torch.cuda.empty_cache()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        return train_losses, valid_losses


def compute_loss_logit(trainer: CodeSimilarityTrainer, batched_data):
    """Loss strategy for fine tuning BERT using the pooler output."""
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


def compute_loss_SBERT_logit(trainer: CodeSimilarityTrainer, batched_data):
    """Loss strategy for fine tuning BERT using the pooler output."""
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


def compute_loss_SBERT_triplet(trainer: CodeSimilarityTrainer, batched_data):
    """Loss strategy for fine tuning BERT using the pooler output."""
    encs_a, encs_p, encs_n = batched_data
    # Converting to cuda tensors if needed
    put_batch_encoding_to_device(encs_a, trainer.device)
    put_batch_encoding_to_device(encs_p, trainer.device)
    put_batch_encoding_to_device(encs_n, trainer.device)
    embs_a = trainer.model(encs_a)  # anchor
    embs_p = trainer.model(encs_p)  # positive
    embs_n = trainer.model(encs_n)  # negative
    return trainer.loss_func(embs_a, embs_p, embs_n)


def compute_loss_contrastive(trainer: CodeSimilarityTrainer, batched_data):
    """Loss strategy for fine tuning BERT using the pooler output."""
    if isinstance(trainer.loss_func, losses.SelfSupervisedLoss):
        ref_input, aug_input = batched_data
        ref_emb = trainer.model(ref_input)
        aug_emb = trainer.model(aug_input)
        return trainer.loss_func(ref_emb, aug_emb)
    else:
        inputs, labels = batched_data
        embeddings = trainer.model(inputs)
        return trainer.loss_func(embeddings, labels)
