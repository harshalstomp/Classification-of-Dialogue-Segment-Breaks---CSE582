import torch
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import json
os.environ['HF_HOME'] = "/scratch/hmp5565/"
os.environ['HF_HUB_CACHE'] = "/scratch/hmp5565/"
#torch.backends.cuda.max_split_size_mb = 100
torch.cuda.empty_cache()

def encode_data(data):

    inputs = tokenizer(data['encoded_row'].tolist(),
                       #data['utterance1'].tolist(),
                       #data['utterance2'].tolist(), 
                       #utterance1_texts, utterance2_texts,
                       padding=True, 
                       truncation=True, 
                       return_tensors='pt')
    print(inputs['input_ids'][0])
    print(len(inputs['input_ids'][0]))
    labels = torch.tensor(data['label'].tolist())
    print(labels)
    return inputs.to(device), labels.to(device)
    #return inputs, labels

def evaluate_model(model, inputs, labels):
    torch.cuda.empty_cache() 
    outputs = model(**inputs, labels=labels)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    return predictions


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

from transformers import RobertaTokenizer, RobertaForSequenceClassification


#model_name2 = 'bert-base-uncased'
#tokenizer = BertTokenizer.from_pretrained(model_name2)
#tokenizer = AutoTokenizer.from_pretrained('gpt2')
model_name = 'roberta-base'  # You can use other RoBERTa variants like 'roberta-large'

#tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

train_inputs, train_labels = encode_data(train_data)
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 12

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()
    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")

# Encode test data
test_inputs, test_labels = encode_data(test_data)

# Perform classification
with torch.no_grad():
    model.eval()
    predictions = evaluate_model(model, test_inputs, test_labels)

predictions = predictions.tolist()
df_class_report = classification_report(test_labels.tolist(), predictions)
#df = pd.DataFrame.from_dict(df_class_report)
print(df_class_report)
