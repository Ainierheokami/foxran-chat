model_path="/workspace/foxran-chat/models/Baichuan2-7B-Base"
lora_path="/workspace/foxran-chat/lora/fox_base"

python fox_run.py \
    --model $model_path \
    --lora $lora_path \
    --bit4