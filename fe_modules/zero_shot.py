from transformers import pipeline

import torch


class ZeroShot:
    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device: str = "cuda"):
        self.device = device
        # self.model = pipeline("zero-shot-classification", model=model_name, device=torch.device(self.device))
        self.model = pipeline("zero-shot-classification", model=model_name)

    def predict_proba(self, texts: list, labels: list, batch_size: int = 128, multi_label: bool = False):
        output = self.model(texts, labels, multi_label=multi_label)
        return output

    def __call__(self, texts: list, labels: list, batch_size: int = 128, multi_label: bool = False):
        return self.predict_proba(texts, labels, batch_size, multi_label)

# import torch
# from torch.utils.data import TensorDataset, DataLoader
#
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# from tqdm.auto import tqdm
#
#
# class ZeroShot:
#     def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device: str = "cuda"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
#         self.device = device
#         self.model.to(self.device)
#         self.model = self.model.eval()
#
#     def predict_proba(self, texts: list, labels: list, batch_size: int = 128,
#                       multi_label: bool = False) -> torch.Tensor:
#         tokenized_inputs = self.tokenizer(texts, labels, truncation=True, padding=True, return_tensors="pt")
#         input_ids = tokenized_inputs["input_ids"]
#
#         dataset = TensorDataset(input_ids)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
#         outputs = list()
#
#         for batch in tqdm(dataloader):
#             b_inputs = batch
#             b_inputs = b_inputs.to(self.device)
#
#             with torch.inference_mode():
#                 output = self.model(b_inputs, multi_label=multi_label)
#                 prediction = torch.softmax(output["logits"][0], -1).tolist()
#                 outputs.append(prediction)
#
#         output = torch.cat(outputs, dim=0)
#
#         return output
#
#     def __call__(self, texts: list, labels: list, batch_size: int = 128, multi_label: bool = False) -> torch.Tensor:
#         return self.predict_proba(texts, labels, batch_size, multi_label)
