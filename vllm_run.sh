# model_path="/workspace/foxran-chat/models/Baichuan2-7B-awq"

# python fox_run.py \
#     --model $model_path \
#     --use_vllm

# python -m vllm.entrypoints.openai.api_server --model $model_path --host "0.0.0.0" --port 5000 --quantization "awq" --trust-remote-code

# xinference-local --host 0.0.0.0 --port 5000

# /workspace/llama.cpp/main -m /workspace/foxran-chat/models/Baichuan2-13B-awq/model_q4_K_M.gguf --repeat-penalty 1.1 -cml -ngl 40 --ctx-size 2048

export OLLAMA_HOST=0.0.0.0:5300
ollama serve