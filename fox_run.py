# 导入包
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datasets import load_dataset
import transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, AutoConfig

from accelerate import init_empty_weights
# from peft import (
#     PeftModel,
#     LoraConfig,
#     get_peft_model,
#     prepare_model_for_kbit_training,
#     set_peft_model_state_dict,
# )
from peft.peft_model import PeftModel
# from peft.config import PeftConfig

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--lora", type=str, default="", required=False)
parser.add_argument("--bit8", action='store_true', default=False)
parser.add_argument("--bit4", action='store_true', default=False)
parser.add_argument("--fastllm", action='store_true', default=False)
args = parser.parse_args()



BASE_MODEL: str = args.model
LORA_WEIGHTS: str = args.lora

USE_FASTLLM: bool = args.fastllm
LOAD_4BIT: bool = args.bit4
LOAD_8BIT: bool = args.bit8

print(f'torch.cuda.is_available -> {torch.cuda.is_available()}')

# 获取显存
def get_max_memory_dict():
    suggestion = f'{int(torch.cuda.mem_get_info()[0]/1024**3)}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {}
    max_memory:dict = {i: suggestion for i in range(n_gpus)}
    # max_memory['cpu'] = f'{32}GiB' # Add CPU memory limit here

    return max_memory if len(max_memory) > 0 else None
max_memory = get_max_memory_dict()
print('Max_memory设置', max_memory)

# 载入模型
def hf_load():
    print('try to load model...')
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        # llm_int8_enable_fp32_cpu_offload=True,
    )
    bit8_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # base_model_params
    params:dict = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "max_memory": max_memory,
        "device_map": "auto",
    }

    if USE_FASTLLM:
        # 内存载入模型进行转换，然后交由fastllm
        params["device_map"] = "cpu"

    elif LOAD_8BIT:
        params["quantization_config"] = bit8_config

    elif LOAD_4BIT:
        params["quantization_config"] = nf4_config

    config  = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )
    model.tie_weights()

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        **params,
    )

    if LORA_WEIGHTS:
        print("try to load Lora...")
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            trust_remote_code=True,
        )
        print('load lora is ok.')

    print("load model finish.")
    return model
    
model = hf_load()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

if USE_FASTLLM:
    # need install fastllm 
    from fastllm_pytools import llm
    
    if LOAD_4BIT:
        model = llm.from_hf(model, tokenizer, dtype = "int4")
    elif LOAD_8BIT:
        model = llm.from_hf(model, tokenizer, dtype = "int8")
    else:
        model = llm.from_hf(model, tokenizer)
        
else:
    # model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    

import gradio as gr
import platform
import traceback

from transformers import TextIteratorStreamer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    # print(Fore.YELLOW + Style.BRIGHT + "欢迎使用百川大模型，输入进行对话，vim 多行输入，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。")
    return []

template = dict(
    SYSTEM='<|System|>:{system}\n',
    INSTRUCTION='<|User|>:{input}<eoh>\n<|Bot|>:'
)

system_template = '你是一只可爱的小狐狸，请以可爱的形式回复消息'
message = '你好'

history = clear_screen()

def evaluate(
    prompt,
    # history,
    # input,
    temperature=0.1,
    top_p=0.45,
    top_k=80,
    num_beams=1,
    max_new_tokens=1024,
    repetition_penalty=1.1,
    **kwargs,
):
    # prompt = template['SYSTEM'].format(system=system_template) + template["INSTRUCTION"].format(input=prompt)
    print('###################')
    print('Prompt内容：\n', prompt)
    
    generation_config = GenerationConfig.from_pretrained(
        BASE_MODEL,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_time= 60,
        max_new_tokens=max_new_tokens,
        # repetition_penalty=repetition_penalty,
        **kwargs,
    )
    
    # print(generation_config)
    fastllm_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_beams": num_beams,
        "do_sample": True,
        "max_time": 60,
        **kwargs,
    }
    
    # history.append({"role": "user", "content": prompt})
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    streamer = TextIteratorStreamer(tokenizer) # type: ignore
    
    try:
        if USE_FASTLLM:
            # 流式传输：
            output = []
            for response in model.stream_chat(tokenizer, prompt, **fastllm_config):
                output.append(response)
                print(response, flush = True, end = "")
            output = "".join(output)

            # # 普通Chat：
            # output = model.chat(tokenizer, now_instruction, **fastllm_config)
            # print(output)
        else:
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids.cuda(),
                    generation_config=generation_config,
                    repetition_penalty=float(repetition_penalty),
                    streamer=streamer
                )
            input_ids_len = input_ids.size(1)
            response_ids = generation_output[:, input_ids_len:].cpu()
            output = "".join(tokenizer.batch_decode(response_ids))
        
        print('output', output)
        print('###################\n')
    
        # output = "".join(output)
        


        return [(prompt, output)]
    
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        print("错误：输入的指令无法生成回复，请重新输入。") 

chatbot = gr.Chatbot()
demo = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="在此输入任务/指令/历史", value=template['SYSTEM'].format(system=system_template) + template["INSTRUCTION"].format(input=message)
        ),
        # gr.components.Textbox(
        #     lines=2, label="History", placeholder="这里输入历史记录"
        # ),
        # gr.components.Textbox(
        #     lines=2, label="Input", placeholder="这里输入input"
        # ),
        gr.components.Slider(minimum=0, maximum=1, value=0.45, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.9, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=80, label="Top k"),
        gr.components.Slider(minimum=1, maximum=5, step=1, value=1, label="Beams"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=1024, label="Max new tokens"
        ),
        gr.components.Slider(
            minimum=0.1, maximum=10.0, step=0.1, value=1.1, label="Repetition Penalty"
        ),
        # gr.components.Slider(
        #     minimum=0, maximum=2000, step=1, value=256, label="Max memory"
        # ),
    ],
    outputs=[chatbot],
    allow_flagging="auto",
    title="🦊🦊🦊自用的超简易狐妖繎chat端🦊🦊🦊",
    description="🦊🦊🦊🦊🦊🦊",
)
demo.queue().launch(server_name="0.0.0.0",server_port=5000,inbrowser=True)