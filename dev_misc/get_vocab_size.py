from transformers import AutoTokenizer

models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen3-0.6B",
]

for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(model, len(tokenizer.get_vocab()))