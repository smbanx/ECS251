import os

# set visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from transformers import pipeline
from datasets import load_dataset

device = 0 if torch.cuda.is_available() else -1
sentiment_pipeline = pipeline("text-classification",model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
ds = load_dataset("imdb")

temp = []

for i in ds["train"]['text'][0:100]:
  temp.append(sentiment_pipeline(i[:512])[0])

