# Reward Model Project Summary

## 🎯 Goal
To train a simple reward model that captures human preferences using a small dataset of prompts and ranked responses.

---

## 🧠 Methodology

1. **Data Preparation**
    - 1 prompt was used: *"Write a short joke about AI."*
    - 4 candidate answers were manually ranked from 1 (best) to 4 (worst).
    - The CSV format: `prompt, answer, rank`

2. **Model**
    - Base: `distilbert-base-uncased`
    - Head: Regression layer (`num_labels=1`) to predict a reward score
    - Tokenizer and model saved locally to `q2_reward/model/`

3. **Training**
    - Optimizer: `AdamW`
    - Epochs: 3
    - Batch size: 4
    - Device: GPU (if available) or CPU

4. **Evaluation**
    - Tested on 4 new AI-related jokes (unseen during training)
    - Reward scores were generated using the trained model

---

## 📊 Evaluation Results

The model produced the following scores:

| Answer | Score |
|--------|-------|
| Why did the AI write a poem? To process its feelings in binary. | 2.2381 |
| The AI got a job as a comedian. Nobody laughed.                 | 2.3027 |
| AI went to therapy. It needed to sort its neural issues.       | 2.2385 |
| An AI tells jokes... but only other AIs understand them.       | 2.1392 |

> ✅ Scores reflect the model's learned preference pattern from the training examples.
