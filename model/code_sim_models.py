import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
#from pytorch_metric_learning import losses

# Hugging Face Transformers (CodeBERT etc.)
import transformers
from transformers import AutoTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling

# Libraries for logging
from tqdm.auto import tqdm

from typing import Callable, Iterable, Protocol, Tuple

# NOTE: Pooling strategy returns a pooled output based on the output of a transformer, this could be the:
# - `pooler_output` which is the embedding of the CLS token in BERT models,
# - a mean pooled version of the last hidden states
# - a max pooled version of the last hidden states
# - an attention pooled version of the last hidden states
PoolingStrategy = Callable[[BaseModelOutputWithPooling, torch.Tensor], torch.Tensor]


def cls_pooling_strat(output: BaseModelOutputWithPooling, *_):
    return output.pooler_output


def max_pooling_strat(output: BaseModelOutputWithPooling, mask: torch.Tensor):
    pooled_output = output.last_hidden_state * mask  # mask out irrelevant embeddings
    return pooled_output.max(dim=1).values


def mean_pooling_strat(output: BaseModelOutputWithPooling, mask: torch.Tensor):
    pooled_output = output.last_hidden_state * mask  # mask out irrelevant embeddings
    return pooled_output.mean(dim=1)


class AttentionPooler(nn.Module):
    def __init__(self, encoder_dim, attention_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(encoder_dim,
                      attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, output: BaseModelOutputWithPooling, mask: torch.Tensor):
        mask = mask.squeeze(-1)  # remove unnecessary 1 dim
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
        mask = inputs['attention_mask'].unsqueeze(-1)  # Unsqueeze for broadcasting
        # Pass through BERT
        output: BaseModelOutputWithPooling = self.bert(**inputs)
        # Pool output
        pooled_output = self.pooling_strat(output, mask)
        # Classification layer
        logits = self.cls(self.drop(pooled_output))
        return logits
    
    @torch.no_grad()
    def predict(self, code_a: str|Iterable[str], code_b: str|Iterable[str]):
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
        probs = torch.sigmoid(logits.squeeze(-1))
        return probs


class CodeSimSBertTripletENC(nn.Module):
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
        # Pool the BERT output
        pooled_output = self.drop(self.pooling_strat(output, mask))
        return pooled_output

    @torch.no_grad()
    def predict(self, code_a: str|Iterable[str], code_b: str|Iterable[str]):
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
        
        distance_function = lambda x, y: 1 - F.cosine_similarity(x, y)
        # Calculate the pairwise cosine distances
        mid = len(codes)//2
        dst = distance_function(outputs[:mid,:], outputs[mid:,:])
        return dst/2 # This is inside [0,1]


class CodeSimSBertTripletCLS(nn.Module, SimilarityClassifier):
    def __init__(self, embedding_size, hidden_sizes=(512,256), num_classes=2, dropout=0.2):
        super().__init__()
        self.embedding_size = embedding_size
        # Size of concatenated emb1, emb2, distance(emb1, emb2)
        input_size = 2*embedding_size+1
        # Classifier head
        self.cls_head = nn.Sequential(
            nn.Linear(input_size,
                      hidden_sizes[0]),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0],
                      hidden_sizes[1]),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1],
                      num_classes),
        )
    
    def forward(self, emb_1, emb_2) -> torch.Tensor:
        d = (1 - F.cosine_similarity(emb_1, emb_2, dim=1)).unsqueeze(dim=1)  # cosine distance
        h = torch.cat([emb_1, emb_2, d], dim=1)  # input feature vector
        logits = self.cls_head(h)
        return logits

    @torch.no_grad()
    def predict(self, code_a: str|Iterable[str], code_b: str|Iterable[str]):
        raise NotImplementedError()


class CodeSimSBertLinearCLS(nn.Module, SimilarityClassifier):
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
        # Pool the BERT output
        pooled_u = self.pooling_strat(u, mask_u)
        pooled_v = self.pooling_strat(v, mask_v)
        # Construct the feature vector [ u, v, |u - v| ]
        h = torch.cat([pooled_u, pooled_v, torch.abs(pooled_u - pooled_v)], dim=1)
        # Classification layer
        logits = self.cls(self.drop(h))
        return logits

    @torch.no_grad()
    def predict(self, code_a: str|Iterable[str], code_b: str|Iterable[str]):
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
        probs = torch.sigmoid(logits.squeeze(-1))
        return probs


class CodeSimCombinedModel(nn.Module, SimilarityClassifier):
    def __init__(
        self,
        bert: transformers.BertModel,  # BERT based model instance
        freeze_bert=False,
        pooling_strat: PoolingStrategy = cls_pooling_strat,
        emb_head_hidden_sizes = (512, 256),
        emb_head_output_size = 256,
        cls_head_hidden_sizes = (512, 256),
        cls_head_num_classes = 2,
        dropout_rate=0.2,
    ):
        super().__init__()
        
        if freeze_bert: freeze_model(bert)
        self.bert = bert
        self.bert_tokenizer = None
        self.pooling_strat = pooling_strat
        
        # Nonlinearity
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

        encoder_dim = bert.config.hidden_size
        # Projection head for contrastive learning task (for meaningfull embeddings)
        self.emb_head = self._create_mlp(encoder_dim, *emb_head_hidden_sizes, emb_head_output_size)
        # Classification head for multiclass tasks
        # NOTE: input size is 3*encoder_dim, see SBERT classification method for explanation
        self.cls_head = self._create_mlp(3*encoder_dim, *cls_head_hidden_sizes, cls_head_num_classes)

    def _create_mlp(self, in_feats, hidden_size_1, hidden_size_2, out_feats):
        return nn.Sequential(
            nn.Linear(in_feats,
                      hidden_size_1),
            self.relu,
            self.drop,
            nn.Linear(hidden_size_1,
                      hidden_size_2),
            self.relu,
            self.drop,
            nn.Linear(hidden_size_2,
                      out_feats),
        )

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        mask = inputs['attention_mask'].unsqueeze(-1)  # Unsqueeze for broadcasting
        # Pass through BERT
        output = self.bert(**inputs)
        # Pool the BERT output
        output = self.drop(self.pooling_strat(output, mask))
        return output

    def forward_train(self, inputs: BatchEncoding, use_proj=True):
        """NOTE: Expects stacked a,p,n inputs."""
        # Pass through embedding projection head layers if needed.
        # This is the default behavior, used for training, because
        # loss is calculated on the projection head's dimension.
        pooled_output = self.forward(inputs)
        emb_head_output = self.emb_head(pooled_output) if use_proj else pooled_output
        # Split embeddings into anchor, positive and negative
        pooled_a, pooled_p, pooled_n = pooled_output.split(pooled_output.shape[0]//3)
        # Input feature vectors for classifier head
        h_pos = torch.cat([pooled_a, pooled_p, torch.abs(pooled_a - pooled_p)], dim=1)
        h_neg = torch.cat([pooled_a, pooled_n, torch.abs(pooled_a - pooled_n)], dim=1)
        # Logits
        cls_head_output = self.cls_head(torch.cat([h_pos, h_neg], dim=0))
        return emb_head_output, cls_head_output

    def predict(self, code_a: str|Iterable[str], code_b: str|Iterable[str]):
        raise NotImplementedError()


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
        assert len(loaders) == 2, "Please provide the training and validation loaders!"
        self.train_loader = loaders[0]
        self.valid_loader = loaders[1]
        self.loss_func = loss_func
        self.loss_hook = loss_hook
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = GradScaler(self.device)
    
    def train_step(self, iters_to_accumulate: int, print_every: int):
        """Train one epoch."""
        self.model.train()
        
        sum_loss = 0.0
        num_iter = len(self.train_loader)
        
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
        
        # Return the average training loss
        avg_loss = sum_loss / num_iter
        return avg_loss
    
    @torch.no_grad()
    def valid_step(self):
        """Evaluate the model on validation data."""
        # Set the model to evaluation mode
        self.model.eval()
        
        sum_loss = 0.0
        num_iter = len(self.valid_loader)
        for data in tqdm(self.valid_loader):
            loss = self.loss_hook(self, data)
            sum_loss += loss.item()
        
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
            train_loss = self.train_step(iters_to_accumulate, print_every)
            valid_loss = self.valid_step()
            print(f"EPOCH {epoch + 1}/{epochs} complete. AVG loss: {train_loss}, AVG validation loss: {valid_loss}")
            
            if valid_loss < best_loss:
                print(f"Best validation loss improved from {best_loss} to {valid_loss}.")
                best_loss = valid_loss
                torch.save(self.model.state_dict(), BEST_MODEL_PATH)
            
            torch.cuda.empty_cache()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        
        return train_losses, valid_losses
    
    # NOTE: Trains a classifier model with training data for triplet models
    def train_cls_head(self, epochs: int, lr=1e-3, weight_decay=0):
        if not isinstance(self.model, CodeSimSBertTripletENC):
            raise ValueError(f"Model needs to be of type: {CodeSimSBertTripletENC.__name__}")
        
        self.model.eval()
        cls_model = CodeSimSBertTripletCLS(self.model.bert.config.hidden_size).to(self.device)
        cls_optim = torch.optim.Adam(cls_model.parameters(), lr=lr, weight_decay=weight_decay)
        cls_loss = nn.CrossEntropyLoss()
        
        print("Training triplet classifier head (CLS).")
        cls_model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            print(f'CLS EPOCH {epoch + 1}/{epochs}')
            for iter, data in enumerate(tqdm(self.train_loader)):
                encs_a, encs_p, encs_n = data
                
                batch_size = encs_a["input_ids"].shape[0]
                
                put_batch_encoding_to_device(encs_a, self.device)
                put_batch_encoding_to_device(encs_p, self.device)
                put_batch_encoding_to_device(encs_n, self.device)
                
                embs_a = self.model(encs_a)  # anchor
                embs_p = self.model(encs_p)  # positive
                embs_n = self.model(encs_n)  # negative
                
                logits_p = cls_model.forward(embs_a, embs_p); labels_p = torch.full((batch_size,),1)
                logits_n = cls_model.forward(embs_a, embs_n); labels_n = torch.full((batch_size,),0)
                logits = torch.cat([logits_p, logits_n], dim=0)
                labels = torch.cat([labels_p, labels_n], dim=0)
                
                loss = cls_loss(logits, labels.to(self.device))
                loss.backward()
                cls_optim.step()
                total_loss += loss.item()
            print(f"CLS EPOCH {epoch + 1}/{epochs} complete. AVG loss: {total_loss/len(self.train_loader)}")
        return cls_model


def Create_CombinedLoss(w_emb=1.0, w_cls=1.0, margin=1.0, distance_function=None):
    # Use cosine distance as a distance function
    if distance_function is None: distance_function = lambda x, y: 1 - F.cosine_similarity(x, y)
    
    emb_loss = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=margin)
    cls_loss = nn.CrossEntropyLoss()
    
    def combined_loss(emb_a, emb_p, emb_n, logits, labels):
        return w_emb * emb_loss(emb_a, emb_p, emb_n) + w_cls * cls_loss(logits, labels)
    return combined_loss


def compute_loss_logit(trainer: CodeSimilarityTrainer, batched_data):
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

# TODO: maybe stack the inputs
def compute_loss_triplet(trainer: CodeSimilarityTrainer, batched_data):
    """Loss strategy for finetuning BERT."""
    encs_a, encs_p, encs_n = batched_data
    # Converting to cuda tensors if needed
    put_batch_encoding_to_device(encs_a, trainer.device)
    put_batch_encoding_to_device(encs_p, trainer.device)
    put_batch_encoding_to_device(encs_n, trainer.device)
    embs_a = trainer.model(encs_a)  # anchor
    embs_p = trainer.model(encs_p)  # positive
    embs_n = trainer.model(encs_n)  # negative
    return trainer.loss_func(embs_a, embs_p, embs_n)


def compute_loss_combined(trainer: CodeSimilarityTrainer, batched_data):
    """Loss strategy for finetuning BERT."""
    encs_a, encs_p, encs_n = batched_data
    batch_size = encs_a["input_ids"].shape[0]
    # create labels for binary classification
    labels_p = torch.full((batch_size,),1)
    labels_n = torch.full((batch_size,),0)
    labels = torch.cat([labels_p,labels_n], dim=0)
    inputs = { key: torch.cat([encs_a[key], encs_p[key], encs_n[key]]) for key in encs_a.keys()}
    put_batch_encoding_to_device(inputs, trainer.device)
    embs, logits = trainer.model.forward_train(inputs)
    embs_a, embs_p, embs_n = embs.split(embs.shape[0]//3)
    return trainer.loss_func(embs_a, embs_p, embs_n, logits, labels.to(trainer.device))
