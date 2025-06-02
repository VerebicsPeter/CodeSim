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
    CodeSimSBertTripletCLS,
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
    model3_1 = CodeSimSBertTripletENC(bert).to(DEVICE)
    model3_2 = CodeSimSBertTripletCLS(bert.config.hidden_size).to(DEVICE)
    model4 = CodeSimCombinedModel(bert).to(DEVICE)

    emb1 = model1(inputs)
    print(f"{model1.__class__.__name__} output shape:", emb1.shape)

    emb2 = model2(inputs, inputs)
    print(f"{model2.__class__.__name__} output shape:", emb2.shape)

    emb3 = model3_1(inputs)
    print(f"{model3_1.__class__.__name__} ENC output shape:", emb3.shape)
    out3 = model3_2(emb3, emb3)
    print(f"{model3_2.__class__.__name__} CLS output shape:", out3.shape)

    emb4 = model4(inputs)
    print(f"{model4.__class__.__name__} output shape:", emb4.shape)
    
    apn_inputs = { 
        key: torch.cat([inputs[key], inputs_p[key], inputs_n[key]]) for key in inputs.keys()
    }
    t_es, t_ls = model4.forward_train(apn_inputs)
    print(f"{model4.__class__.__name__} output shape for train pass:", t_es.shape, t_ls.shape)


def count_overlapping_pids(
    seed_value, factory,
    pretrained_bert_name="huggingface/CodeBERTa-small-v1",
    num_rows=50_000,
):
    set_seed(seed_value=seed_value)
    # Dataset creation
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    dataset = factory(tokenizer=tokenizer, num_rows=num_rows)
    
    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    print(len(train_data), len(valid_data))
    
    # Access problem IDs from the splits
    train_pids = {dataset.pids[i] for i in train_data.indices}
    valid_pids = {dataset.pids[i] for i in valid_data.indices}
    valid_pids_l = [dataset.pids[i] for i in valid_data.indices]    
    
    # Count overlaps
    overlap_pids = train_pids.intersection(valid_pids)
    print(f"Count of problem IDs:\n{len(train_pids.union(valid_pids))}")
    print(f"Count of overlapping problem IDs between splits:\n{len(overlap_pids)}")
    num_overlapping_valid_rows = sum(1 for pid in valid_pids_l if pid in train_pids)
    print(f"Count of overlapping rows in validation split:\n{num_overlapping_valid_rows}")
    print(f"Ratio of overlapping rows in validation split:\n{num_overlapping_valid_rows/len(valid_data)}")
