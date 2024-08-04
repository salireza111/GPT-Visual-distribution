from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def predict_next_token_distribution(text, top_k=10):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    last_token_logits = logits[0, -1, :]
    probabilities = torch.softmax(last_token_logits, dim=-1).cpu().numpy()
    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_k_probabilities = probabilities[top_k_indices]
    top_k_tokens = [tokenizer.decode([token]) for token in top_k_indices]

    return top_k_tokens, top_k_probabilities

text = "ligands have been used to"
top_k_tokens, top_k_probabilities = predict_next_token_distribution(text)

for token, prob in zip(top_k_tokens, top_k_probabilities):
    print(f"Token: {token}, Probability: {prob:.4f}")

plt.figure(figsize=(10, 5))
plt.bar(top_k_tokens, top_k_probabilities)
plt.xlabel('Tokens')
plt.ylabel('Probabilities')
plt.title('Top K Next Token Predictions')
plt.show()