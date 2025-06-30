import torch
import torch.nn.functional as F
import transformers
import gradio as gr
from collections import OrderedDict

import os
import sys
root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(root_dir)

try:
    from model import train
    from utils import splitter
except Exception as error:
    print(error)


LANGUAGE_MAP = {"Python": "python", "C": 'c', "C++": "cpp"}
CHECKPOINT = "huggingface/CodeBERTa-small-v1"
FINETUNES = {
    "CodeBERTa tuned w/ Triplet Loss": "/home/peter/Downloads/codesim_models/CLS_pooling_new_dataset_8_epochs_1/model.pt",
    "CodeBERTa tuned w/ InfoNCE Loss": "/home/peter/Downloads/codesim_models/CLS_pooling_new_dataset_infoNCE_amplified_1/model.pt",
}

BERT_INST = transformers.AutoModel.from_pretrained(CHECKPOINT)
BERT_TKNR = transformers.AutoTokenizer.from_pretrained(CHECKPOINT)
MODEL_INST = train.CodeSimContrastiveEncoder(BERT_INST)
MODEL_INST.eval()
print("Model set to evaluation mode.")

# NOTE: this is not good for an online app...
__CODE_SUBSEQ_CACHE = {
    "code1": [],
    "code2": [],
}

@torch.no_grad
def embedding_pipeline(codes):
    toks = BERT_TKNR(codes, padding=True, truncation=True, return_tensors="pt")
    return MODEL_INST.forward(toks)


def on_process_code(code1, lang1, code2, lang2):
    # TODO: split code snipepts into semantic parts
    embs = embedding_pipeline([code1, code2])
    cos_sim = F.cosine_similarity(embs[0], embs[1], dim=0)
    cos_sim_str = f"Cosine similarity: {cos_sim}"
    
    code_snips1 = splitter.split_code_by_semantics(code1, LANGUAGE_MAP[lang1])
    code_snips2 = splitter.split_code_by_semantics(code2, LANGUAGE_MAP[lang2])
    
    if not code_snips1 or not code_snips2:
        return cos_sim_str, []
    else:
        __CODE_SUBSEQ_CACHE["code1"] = code_snips1
        __CODE_SUBSEQ_CACHE["code2"] = code_snips2
        embs_snips1 = embedding_pipeline(code_snips1)
        embs_snips2 = embedding_pipeline(code_snips2)
        embs_snips1 = F.normalize(embs_snips1, dim=1)
        embs_snips2 = F.normalize(embs_snips2, dim=1)
        sim_mat = embs_snips1 @ embs_snips2.T
        sim_mat = sim_mat.detach().cpu().numpy()
        return cos_sim_str, sim_mat


def on_select_sim_mat_cell(evt: gr.SelectData):
    code1 = __CODE_SUBSEQ_CACHE["code1"][evt.index[0]]
    code2 = __CODE_SUBSEQ_CACHE["code2"][evt.index[1]]
    return (f"You selected {evt.value} at {evt.index} from {evt.target}", code1, code2)


def on_load_checkpoint(name: str):
    if name not in FINETUNES: raise ValueError("Fine-tuned model does not exist!")
    
    path = FINETUNES[name]
    
    state = torch.load(path, map_location=torch.device("cpu"))
    state = OrderedDict({name.replace("bert.", "enc_model."): value for name, value in state.items()})
    
    MODEL_INST.load_state_dict(state)
    message = "Model state loaded."
    print( message )
    return message


def app_tab():    
    with gr.Tab("Code Similarity Tool"):
        gr.Markdown("## Code Similarity Tool")
        
        with gr.Row():
            with gr.Column():
                lang_selector1 = gr.Dropdown(
                    list(LANGUAGE_MAP.keys()), value="Python", label="Language for Snippet 1")
                code_input1 = gr.Code(label="Snippet 1", language="python", max_lines=12)
                lang_selector2 = gr.Dropdown(
                    list(LANGUAGE_MAP.keys()), value="Python", label="Language for Snippet 2")
                code_input2 = gr.Code(label="Snippet 2", language="python", max_lines=12)
                submit_btn = gr.Button("Process")
            with gr.Column():
                submit_out = gr.Textbox(label="Output")
                code_subseq1 = gr.Code(label="Snippet 1 - subseq", max_lines=12)
                code_subseq2 = gr.Code(label="Snippet 2 - subseq", max_lines=12)
        
                gr.Markdown("### Similarity Matrix")
                
                table = gr.Dataframe([])
                table_out  = gr.Textbox(label="Output")
                table.select(
                    fn=on_select_sim_mat_cell,
                    inputs=None,
                    outputs=[table_out, code_subseq1, code_subseq2]
                )

        # Update the language of the code editors when the dropdown changes
        lang_selector1.change(
            lambda lang: gr.update(language=LANGUAGE_MAP[lang]),
            inputs=lang_selector1,
            outputs=code_input1
        )
        lang_selector2.change(
            lambda lang: gr.update(language=LANGUAGE_MAP[lang]),
            inputs=lang_selector2,
            outputs=code_input2
        )

        # Process button combines the snippets and displays them
        submit_btn.click(
            fn=on_process_code,
            inputs=[code_input1, lang_selector1, code_input2, lang_selector2],
            outputs=[submit_out, table]
        )


def opt_tab():
    with gr.Tab("Options"):
        gr.Markdown("## Load Models")
        dropdown = gr.Dropdown(
            choices=[
                "CodeBERTa base",
                "CodeBERTa tuned w/ Triplet Loss",
                "CodeBERTa tuned w/ InfoNCE Loss",
            ],
            label="Choose a model",
        )
        dropdown_out = gr.Textbox(label="Output")
        dropdown.change(
            fn=on_load_checkpoint,
            inputs=dropdown,
            outputs=dropdown_out
        )


def main():
    with gr.Blocks() as demo:
        with gr.Tabs():
            app_tab()
            opt_tab()
    demo.launch()


if __name__ == "__main__":
    main()
