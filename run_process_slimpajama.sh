python scripts/prepare_slimpajama_full.py \
    --source_path data/slimpajama \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-chat-hf  \
    --destination_path data/slimpajama/tokenized/llama2/train \
    --split train \
    --percentage 1.0


python scripts/prepare_slimpajama_full.py \
    --source_path data/slimpajama \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-chat-hf  \
    --destination_path data/slimpajama/tokenized/llama2/validation \
    --split validation \
    --percentage 1.0


python scripts/prepare_slimpajama_full.py \
    --source_path data/slimpajama \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-chat-hf  \
    --destination_path data/slimpajama/tokenized/llama2/test \
    --split test \
    --percentage 1.0