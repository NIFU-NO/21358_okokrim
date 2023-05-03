# Dependencies
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from tabulate import tabulate
from tqdm import trange
import random

# Load model
model = torch.load("tuned_bert_model.pt")

# Load data to be predicted on
pred_df = pd.read_csv("data/pubs_pred.csv")


new_text = pred_df.text.values

# Initiate prediction array
predictions = []

# Run model on each case and add prediction to array
for sentence in new_text:
  test_ids = []
  test_attention_masks = []

  encoding = preprocessing(sentence, tokenizer)
  
  test_ids.append(encoding['input_ids'])
  test_attention_masks.append(encoding['attention_mask'])

  test_ids = torch.cat(test_ids, dim = 0)
  test_attention_masks = torch.cat(test_attention_masks, dim = 0)
  
  with torch.no_grad():
    output = model(
      test_ids.to(device), 
      token_type_ids = None, 
      attention_mask = test_attention_masks.to(device)
      )

  prediction = 'EX' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'IN'
  
  predictions.append(prediction)

# Add predictions to DataFrame
pred_df['BERT_prediction'] = predictions

# Save predictions
pred_df.to_csv("data/bert_predictions2.csv", index = False)
