import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import pandas as pd
import os

class RewardData(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.samples = df.to_dict(orient='records')
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['answer']
        rank = item['rank']
        score = 5 - rank  # higher score = better answer
        tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': torch.tensor([float(score)])
        }

# Load CSV and tokenizer
df = pd.read_csv("../answers.csv")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = RewardData(df, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} Loss:", loss.item())

# Save model
os.makedirs("q2_reward/model", exist_ok=True)
model.save_pretrained("q2_reward/model")
tokenizer.save_pretrained("q2_reward/model")
