import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast


def get_embeddings(texts: list, model_name: str = "setu4993/LaBSE", batch_size: int = 128, device: str = "cuda"):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    english_inputs = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = english_inputs["input_ids"]
    attention_mask = english_inputs["attention_mask"]

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    outputs = []

    for batch in tqdm(dataloader):
        b_inputs, b_attention_mask = batch
        b_inputs = b_inputs.to(device)
        b_attention_mask = b_attention_mask.to(device)

        with torch.no_grad():
            logits = model(b_inputs, b_attention_mask)["pooler_output"]
            outputs.append(logits)

    embeddings = torch.cat(outputs, dim=0)

    return embeddings
