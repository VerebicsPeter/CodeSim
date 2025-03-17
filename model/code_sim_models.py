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

# Libraries for logging
from tqdm.auto import tqdm

from typing import Callable, Iterable, Protocol, Tuple, Any


def freeze_model(model: nn.Module):
    for param in model.parameters(): param.requires_grad = False


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


class FinetunedCodeSimilarityModel(nn.Module, SimilarityClassifier):
    def __init__(
        self,
        bert: transformers.BertModel,  # BERT based model instance
        freeze_bert=False,
        dropout_rate=0.2,
    ):
        super().__init__()
        if freeze_bert: freeze_model(bert)
        self.bert = bert
        self.bert_tokenizer = None
        self.drop = nn.Dropout(dropout_rate)
        self.cls = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        # Pass through BERT
        output = self.bert(**inputs)
        output = output.pooler_output  # Use the pooler output
        output = self.drop(output)
        # Last linear layer
        logits = self.cls(output)
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
        device = self.bert.device
        for k, v in inputs.items(): inputs[k] = v.to(device)
        
        logits = self.forward(inputs)
        
        pred = (torch.sigmoid(logits.squeeze(-1)) > threshold).int()
        return pred


class ContrastiveCodeSimilarityModel(nn.Module, SimilarityClassifier):
    def __init__(
        self,
        # BERT based model instance
        bert: transformers.BertModel,
        freeze_bert=False,
        dropout_rate=0.2,
        # MLP hidden sizes and out features
        mlp_sizes=(512, 256, 128),
        mlp_feat_out=32,
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
        device = self.bert.device
        for k, v in inputs.items(): inputs[k] = v.to(device)
        
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


def compute_loss_finetuned(trainer: CodeSimilarityTrainer, batched_data):
    """Loss strategy for fine tuning BERT using the pooler output."""
    encoding, labels = batched_data
    # Converting to cuda tensors
    for k, v in encoding.items(): encoding[k] = v.to(trainer.device)
    # also convert labels...
    labels = labels.to(trainer.device)
    # Obtaining the logits from the model
    logits = trainer.model(encoding)
    # Computing loss
    loss = trainer.loss_func(logits.squeeze(-1), labels.float())
    return loss


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
