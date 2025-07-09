import torch

from transformers import (
    AutoModel,
    AutoTokenizer,
)
from model.code_sim_models import (
    CodeSimLinearClassifierCross,
    CodeSimLinearClassifierSBert,
    CodeSimContrastiveEncoder,
)
from model.utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_forward_passes(pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1"):
    code = """print("Hello, World!")"""
    bert = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)

    params = {
        "padding":'max_length',  # Pad to max_length
        "max_length": bert_tokenizer.model_max_length,
        "truncation":True,       # Truncate to max_length
        "return_tensors":'pt'    # Return torch.Tensor objects
    }
    inputs = bert_tokenizer(code, **params)
    
    #pooling_strat=code_sim_models.AttentionPooler(768,768)
    model1 = CodeSimLinearClassifierCross(bert).to(DEVICE)
    model2 = CodeSimLinearClassifierSBert(bert).to(DEVICE)
    model3 = CodeSimContrastiveEncoder(bert).to(DEVICE)

    emb1 = model1(inputs)
    print(f"{model1.__class__.__name__} output shape:", emb1.shape)

    emb2 = model2(inputs, inputs)
    print(f"{model2.__class__.__name__} output shape:", emb2.shape)

    emb3 = model3(inputs)
    print(f"{model3.__class__.__name__} output shape:", emb3.shape)
