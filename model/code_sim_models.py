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
from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    get_linear_schedule_with_warmup,
)
from transformers.utils import ModelOutput

# Libraries for logging
from tqdm.auto import tqdm

from typing import Protocol, Callable, Tuple


EmbeddingPipeline = Callable[[BatchEncoding], ModelOutput]


def create_embedding_pipeline(transformer: PreTrainedModel) -> EmbeddingPipeline:
    """Create an embedding function with a `transformer.PreTrainedModel` instance."""
    def pipeline(inputs: BatchEncoding) -> ModelOutput:
        return transformer(**inputs)
    return pipeline


class Trainer(Protocol):
    # Returns train and validation losses in a tuple
    def train(self, epochs: int, **kwargs) -> Tuple: ...


class ContrastiveCodeSimilarityModel(nn.Module):
    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        in_feat=768,  # NOTE: Depends on the embedding pipeline
        mlp_sizes=(512, 256, 128),
        out_feat=32,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.embedding_pipeline = embedding_pipeline

        # Non linearity
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

        assert len(mlp_sizes) == 3, 'MLP must have 3 hidden sizes'

        # MLP 'projection head'
        mlp_layers = []
        mlp_layers.append(nn.Linear(in_feat, mlp_sizes[0]))

        mlp_layers.extend([self.relu, self.drop])
        mlp_layers.append(nn.Linear(mlp_sizes[0], mlp_sizes[1]))

        mlp_layers.extend([self.relu, self.drop])
        mlp_layers.append(nn.Linear(mlp_sizes[1], mlp_sizes[2]))

        mlp_layers.extend([self.relu, self.drop])
        mlp_layers.append(nn.Linear(mlp_sizes[2], out_feat))

        self.mlp = nn.Sequential(*mlp_layers)


    def embed(self, inputs: BatchEncoding) -> torch.Tensor:
        output = self.embedding_pipeline(inputs)
        return output.pooler_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through linear layers
        x = self.mlp(x)
        return x

class ContrastiveCodeSimilarityTrainer(Trainer):
    def __init__(
        self,
        model: ContrastiveCodeSimilarityModel,
        loaders: Tuple[DataLoader, DataLoader],
        loss_func,
        optimizer,
        scheduler = None,
    ):
        self.model = model
        assert len(loaders) == 2, "Please provide train and validation loaders!"
        self.train_loader = loaders[0]
        self.valid_loader = loaders[1]
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.is_self_supervised = isinstance(self.loss_func, losses.SelfSupervisedLoss)
    
    def compute_loss(self, batched_data):
        """Computes the loss value for a batch of data."""
        if self.is_self_supervised:
            ref_input, aug_input = batched_data
            # Transformer
            ref_emb = self.model.embed(ref_input)
            aug_emb = self.model.embed(aug_input)
            # MLP
            ref_emb = self.model(ref_emb)
            aug_emb = self.model(aug_emb)
            
            return self.loss_func(ref_emb, aug_emb)
        else:
            inputs, labels = batched_data
            embeddings = self.model.embed(inputs)
            embeddings = self.model(embeddings)
            
            return self.loss_func(embeddings, labels)
    
    def train_epoch(self):
        """Trains the model for one epoch."""
        N_BATCHES = len(self.train_loader)  # Number of batches
        C_BATCHES = 50                      # Number of batches over which the logged loss is cumulated
        
        def get_last_loss(batch, acc_loss):
            if batch % C_BATCHES == C_BATCHES - 1:
                return 0, acc_loss / C_BATCHES
            if batch == N_BATCHES - 1:
                return 0, acc_loss / (N_BATCHES % C_BATCHES)
            return acc_loss, 0

        # TODO: use a log file
        def log_loss(n_batches, batch, last_loss):
            # Log the average loss over the last  batches
            print('',f'Batch: {batch + 1}/{n_batches}, Loss: {last_loss}')
        
        # Set the model to training mode
        self.model.train()
        sum_loss, acc_loss = 0, 0
        for i, data in tqdm(enumerate(self.train_loader)):
            self.optimizer.zero_grad()
            loss = self.compute_loss(data)
            # Adjust the weights
            loss.backward()
            self.optimizer.step()
            # Increase loss accumulator
            loss_val = loss.item()
            sum_loss += loss_val
            acc_loss += loss_val
            # Update loss accumulator
            acc_loss, last_loss = get_last_loss(N_BATCHES, C_BATCHES, i, acc_loss)
            # Log the loss to the console
            if last_loss:
                log_loss(N_BATCHES, i, last_loss)
        # Return the average loss in the epoch
        avg_loss = sum_loss / N_BATCHES
        return avg_loss
    
    def validate(self):
        """Evaluate the model on validation data."""
        N_BATCHES = len(self.valid_loader)
        # Set the model to evaluation mode
        self.model.eval()
        sum_loss = 0
        with torch.no_grad():
            for data in self.valid_loader:
                loss = self.compute_loss(data)
                sum_loss += loss.item()
        # Return the average evaluation loss
        avg_loss = sum_loss / N_BATCHES
        return avg_loss
    
    def train(self, epochs: int):
        train_losses, valid_losses = [], []
        for epoch in range(epochs):
            print(f'EPOCH {epoch + 1}/{epochs}')
            # Train then validate
            avg_tLoss = self.train_epoch()
            avg_vLoss = self.validate()
            # Adjust the LR scheduler
            if self.scheduler is not None: self.scheduler.step()
            # Log the losses
            print(f"EPOCH {epoch + 1}/{epochs}, AVG loss: {avg_tLoss}, AVG validation loss: {avg_vLoss}")
            train_losses.append(avg_tLoss)
            valid_losses.append(avg_vLoss)
        return train_losses, valid_losses


class BERTCodeSimilarityModel(nn.Module):
    def __init__(
        self,
        bert: transformers.BertModel,  # BERT based model instance
        freeze_bert=False,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.bert = bert
        self.drop = nn.Dropout(dropout_rate)
        self.cls = nn.Linear(self.bert.config.hidden_size, 1)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, bert_input) -> torch.Tensor:
        bert_output = self.bert(**bert_input)
        pooler_output = bert_output.pooler_output
        pooler_output = self.drop(pooler_output)
        logits = self.cls(pooler_output)
        return logits

class BERTCodeSimilarityTrainer(Trainer):
    def __init__(
        self,
        model: BERTCodeSimilarityModel,
        loaders: Tuple[DataLoader, DataLoader],
        loss_func,
        optimizer,
        scheduler,
        device: torch.device
    ):
        self.model = model
        assert len(loaders) == 2, "Please provide train and validation loaders!"
        self.train_loader = loaders[0]
        self.valid_loader = loaders[1]
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = GradScaler('cuda')
    
    def train_epoch(self, iters_to_accumulate: int, print_every: int):
        running_loss = 0.0
        self.model.train()

        num_iters = len(self.train_loader)
        for iter, (encoding, labels) in enumerate(tqdm(self.train_loader)):
            # Converting to cuda tensors
            for k, v in encoding.items(): encoding[k] = v.to(self.device)
            labels = labels.to(self.device)

            # Obtaining the logits from the model
            # Enables autocasting for the forward pass (model + loss)
            with autocast('cuda'):
                # Obtaining the logits from the model
                logits = self.model(encoding)
                # Computing loss
                loss = self.loss_func(logits.squeeze(-1), labels.float())
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
                print(
                    f"Iteration {iter+1}/{num_iters} complete. " +
                    f"Loss: {running_loss / print_every}"
                )
                running_loss = 0.0
    
    def validate(self):
        loss, count = 0,0
        with torch.no_grad():
            for _, (encoding, labels) in enumerate(tqdm(self.valid_loader)):
                # Converting to cuda tensors
                for k, v in encoding.items(): encoding[k] = v.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(encoding)
                loss += self.loss_func(logits.squeeze(-1), labels.float()).item()
                count += 1

        mean_loss = loss / count
        return mean_loss
    
    def train(self, epochs: int, iters_to_accumulate: int = 2):
        BEST_MODEL_PATH = "best_model.pt"  # path to save the best model to
        train_losses, valid_losses = [], []
        best_loss = np.Inf
        
        # print training loss 5 times per epoch
        print_every=len(self.train_loader) // 5
        
        for epoch in range(epochs):
            # Compute training loss
            train_loss = self.train_epoch(iters_to_accumulate, print_every)
            # Compute validation loss
            valid_loss = self.validate()
            print(f"Epoch {epoch+1} complete! Validation Loss: {valid_loss}")
            if valid_loss < best_loss:
                print(f"Best validation loss improved from {best_loss} to {valid_loss}")
                best_loss = valid_loss
                torch.save(self.model.state_dict(), BEST_MODEL_PATH)
            torch.cuda.empty_cache()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        
        return train_losses, valid_losses
