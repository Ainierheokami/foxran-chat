model_path="/workspace/fox-bloom-lora/models/Baichuan2-7B-Base"
lora_path="/workspace/foxran-chat/lora/baichuan2-7b-foxchat-qlora"

python fox_run.py \
    --model $model_path \
    --lora $lora_path \
    --bit4