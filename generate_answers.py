from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import os

# Ensure output folder exists
os.makedirs("q2_reward", exist_ok=True)

# === Config ===
model_name = "gpt2"  # You can change to another model like "tiiuae/falcon-rw-1b" if needed
prompts = [
    "Tell me a joke about robots.",
    "Summarize the plot of 'The Matrix'.",
    "Why is climate change a big deal?",
    "Explain the significance of the moon landing.",
    "What makes a good leader?"
]

# === Load model and tokenizer ===
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # use -1 for CPU, or 0 for GPU if available
)

print("Generating responses...")
rows = []

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    outputs = generator(
        prompt,
        max_length=150,
        num_return_sequences=4,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id  # Prevent warning for padding
    )
    for out in outputs:
        text = out["generated_text"]
        print(f" - {text[:80]}...")  # Preview first 80 chars
        rows.append({"prompt": prompt, "answer": text})

# === Save to CSV ===
df = pd.DataFrame(rows)
df.to_csv("q2_reward/generated_answers_raw.csv", index=False)
print("\nSaved generated answers to q2_reward/generated_answers_raw.csv")
