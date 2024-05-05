import os
import subprocess
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/workspace/foxran-chat/models/Baichuan2-13B-lora-merge'
# quant_path = '/workspace/foxran-chat/models/Baichuan2-13B-awq'
llama_cpp_path = '/workspace/llama.cpp'

# GGUF conversion
print('Converting model to GGUF...')
llama_cpp_method = "q5_0"
convert_cmd_path = os.path.join(llama_cpp_path, "convert-hf-to-gguf.py")
quantize_cmd_path = os.path.join(llama_cpp_path, "quantize")

if not os.path.exists(llama_cpp_path):
    cmd = f"git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_path} && cd {llama_cpp_path} && make LLAMA_CUBLAS=1 LLAMA_CUDA_F16=1"
    subprocess.run([cmd], shell=True, check=True)

subprocess.run([
    f"python {convert_cmd_path} {model_path} --outfile {model_path}/model.gguf"
], shell=True, check=True)

subprocess.run([
    f"{quantize_cmd_path} {model_path}/model.gguf {model_path}/model_{llama_cpp_method}.gguf {llama_cpp_method}"
], shell=True, check=True)