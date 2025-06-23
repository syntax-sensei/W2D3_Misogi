import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

model_dir = os.path.abspath("q2_reward/reward_model/q2_reward/model")  # ensure full local path

# Load model from local directory ONLY
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir, local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, local_files_only=True
)
model.eval()

# Candidate answers
answers = [
    "Why did the AI write a poem? To process its feelings in binary.",
    "The AI got a job as a comedian. Nobody laughed.",
    "AI went to therapy. It needed to sort its neural issues.",
    "An AI tells jokes... but only other AIs understand them."
]

# Score each answer
for i, answer in enumerate(answers):
    tokens = tokenizer(answer, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        output = model(**tokens)
        score = output.logits.item()
    print(f"Answer {i+1} Score: {score:.4f}")
