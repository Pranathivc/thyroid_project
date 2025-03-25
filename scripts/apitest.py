from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Chatbot conversation
chat_history_ids = None
print("Chatbot is ready! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Encode the input and generate response
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = model.generate(
        new_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(chat_history_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Chatbot:", response)