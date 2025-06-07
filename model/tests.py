import torch
from torch.utils.data import (
    random_split,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from model.code_sim_models import (
    CodeSimLinearCLS,
    CodeSimSBertLinearCLS,
    CodeSimSBertTripletENC,
    CodeSimCombinedModel,
)
from model.utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_forward_passes(pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1"):
    code = """print("Hello, World!")"""
    code_p = """hello_str = "Hello, World!"; print(hello_str)"""
    code_n = """def add(x,y): return x+y"""

    bert = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)

    params = {
        "padding":'max_length',  # Pad to max_length
        "max_length": bert_tokenizer.model_max_length,
        "truncation":True,       # Truncate to max_length
        "return_tensors":'pt'    # Return torch.Tensor objects
    }
    inputs = bert_tokenizer(code, **params)
    inputs_p = bert_tokenizer(code_p, **params)
    inputs_n = bert_tokenizer(code_n, **params)
    
    #pooling_strat=code_sim_models.AttentionPooler(768,768)
    model1 = CodeSimLinearCLS(bert).to(DEVICE)
    model2 = CodeSimSBertLinearCLS(bert).to(DEVICE)
    model3 = CodeSimSBertTripletENC(bert).to(DEVICE)
    model4 = CodeSimCombinedModel(bert).to(DEVICE)

    emb1 = model1(inputs)
    print(f"{model1.__class__.__name__} output shape:", emb1.shape)

    emb2 = model2(inputs, inputs)
    print(f"{model2.__class__.__name__} output shape:", emb2.shape)

    emb3 = model3(inputs)
    print(f"{model3.__class__.__name__} output shape:", emb3.shape)

    emb4 = model4(inputs)
    print(f"{model4.__class__.__name__} output shape:", emb4.shape)
    
    apn_inputs = { 
        key: torch.cat([inputs[key], inputs_p[key], inputs_n[key]]) for key in inputs.keys()
    }
    t_es, t_ls = model4.forward_train(apn_inputs)
    print(f"{model4.__class__.__name__} output shape:", t_es.shape, t_ls.shape, "[train pass]")
