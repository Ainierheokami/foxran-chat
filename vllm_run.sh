model_path="/workspace/foxran-chat/models/Baichuan2-7B-awq"

python fox_run.py \
    --model $model_path \
    --use_vllm

# python -m vllm.entrypoints.openai.api_server --model $model_path --host "0.0.0.0" --port 5000 --quantization "awq" --trust-remote-code