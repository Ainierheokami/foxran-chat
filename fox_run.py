# 导入包
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from typing import Any, List, cast

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, AutoConfig
from vllm import LLM, SamplingParams

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
parser.add_argument("--use_fastllm", action='store_true', default=False)
parser.add_argument("--use_vllm", action='store_true', default=False)
args = parser.parse_args()



BASE_MODEL: str = args.model
LORA_WEIGHTS: str = args.lora

USE_FASTLLM: bool = args.use_fastllm
USE_VLLM: bool = args.use_vllm


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
    if USE_VLLM:
        return None,None
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
        "low_cpu_mem_usage": True,
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
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print("load model finish.")
    return model, tokenizer
    
    
import gradio as gr
import platform
import traceback

# from transformers import TextIteratorStreamer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    # print(Fore.YELLOW + Style.BRIGHT + "欢迎使用百川大模型，输入进行对话，vim 多行输入，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。")
    return []

internlm2_chat=dict(
    SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
    INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                    '<|im_start|>assistant\n'),
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>']
)

default_template = dict(
    SYSTEM='<|System|>system\n{system}<|End|>\n',
    INSTRUCTION=('<|User|>user\n{input}<|End|>\n'
                    '<|Bot|>assistant\n'),
    SUFFIX='<|End|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|End|>']
)

template = dict(
    SYSTEM='<|System|>:{system}\n',
    INSTRUCTION='<|User|>:{input}<eoh>\n<|Bot|>:'
)

template = internlm2_chat

system_template = '你是一只可爱的小狐狸，请以可爱的形式回复消息'
message = '你好'

history = clear_screen()

model,tokenizer = hf_load()

if USE_FASTLLM:
    # need install fastllm 
    from fastllm_pytools import llm
    
    if LOAD_4BIT:
        model = llm.from_hf(model, tokenizer, dtype = "int4")
    elif LOAD_8BIT:
        model = llm.from_hf(model, tokenizer, dtype = "int8")
    else:
        model = llm.from_hf(model, tokenizer)
elif USE_VLLM:
    model = LLM(
        model=BASE_MODEL, 
        quantization="awq", 
        trust_remote_code=True,  
        # gpu_memory_utilization=0.95, 
        # max_model_len=768,
        # enforce_eager=True,
        )
else:
    # model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

def evaluate(
    prompt,
    stop_word="",
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

    try:
        if USE_VLLM:
            sampling_params = SamplingParams(
                temperature=temperature, 
                top_p=top_p,
                top_k=top_k,
                stop=cast(list, template.get('STOP_WORDS', [])) + stop_word.split(',') if "," in stop_word else [stop_word],
                max_tokens=max_new_tokens,
                skip_special_tokens=False,
                repetition_penalty=repetition_penalty,
                use_beam_search=False if num_beams == 1 else True,
                best_of=num_beams if num_beams != 1 else None,         
                )
            assert isinstance(model, LLM)
            outputs = model.generate(prompt, sampling_params)
            for i in outputs:
                output = i.outputs[0].text

        else:
            assert isinstance(model, PeftModel) and tokenizer
            gen_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": max_new_tokens,
                "max_time": 30,
                "do_sample": temperature > 0,
                **kwargs,
            }

            if os.path.isfile(os.path.join(BASE_MODEL, "generation_config.json")):
                generation_config = GenerationConfig.from_pretrained(
                    BASE_MODEL,
                    **gen_config,
                )
            else:
                generation_config = GenerationConfig(**gen_config)
            
            # history.append({"role": "user", "content": prompt})
            
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            from transformers.generation.streamers import TextStreamer
            from transformers import StoppingCriteria, StoppingCriteriaList

            class StopWordStoppingCriteria(StoppingCriteria):
                """StopWord stopping criteria."""

                def __init__(self, tokenizer, stop_word):
                    self.tokenizer = tokenizer
                    self.stop_word = stop_word
                    self.length = len(self.stop_word)

                def __call__(self, input_ids, *args, **kwargs) -> bool:
                    cur_text = self.tokenizer.decode(input_ids[0])
                    cur_text = cur_text.replace('\r', '').replace('\n', '')
                    return cur_text[-self.length:] == self.stop_word

            streamer = TextStreamer(tokenizer, skip_prompt=True)  # type: ignore

            def get_stop_criteria(tokenizer, stop_words=[]):
                stop_criteria = StoppingCriteriaList()
                for word in stop_words:
                    stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
                return stop_criteria
            
            stop_words:list[str] = [stop_word]
            for word in cast(list, template.get('STOP_WORDS', [])):
                stop_words.append(word)
            sep = template.get('SEP', '')
            stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=stop_words)
            if USE_FASTLLM:
                # 流式传输：
                output = []
                for response in model.stream_chat(tokenizer, prompt, **gen_config):
                    output.append(response)
                    print(response, flush = True, end = "")
                output = "".join(output)

                # # 普通Chat：
                # output = model.chat(tokenizer, now_instruction, **fastllm_config)
                # print(output)
            else:
                with torch.no_grad():
                    generation_output = model.generate(
                        input_ids=input_ids.cuda(), # type: ignore
                        generation_config=generation_config,
                        repetition_penalty=float(repetition_penalty),
                        streamer=streamer,
                        stopping_criteria=stop_criteria,
                    )
                input_ids_len = input_ids.size(1) # type: ignore
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


if __name__ == "__main__":
    chatbot = gr.Chatbot()
    demo = gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Instruction", placeholder="在此输入任务/指令/历史", value=template['SYSTEM'].format(system=system_template) + template["INSTRUCTION"].format(input=message)  # type: ignore
            ),
            gr.components.Textbox(
                lines=1, label="Stop Words", placeholder="在此输入停止词", value=""
            ),
            # gr.components.Textbox(
            #     lines=2, label="History", placeholder="这里输入历史记录"
            # ),
            # gr.components.Textbox(
            #     lines=2, label="Input", placeholder="这里输入input"
            # ),
            gr.components.Slider(minimum=0, maximum=1, value=0.45, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.9, label="Top p"),
            gr.components.Slider(minimum=-1, maximum=100, step=1, value=80, label="Top k"),
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