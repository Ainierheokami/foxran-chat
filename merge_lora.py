import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import BitsAndBytesConfig


# Ref: https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py

def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # quantization_config=nf4_config,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    if args.llama:
        base_tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    else:
        base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        device_map="cpu",
        # torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--llama", action="store_true", required=False)

    args = parser.parse_args()

    apply_lora(args.model_name_or_path, args.output_path, args.lora_path)

# use eg: python merge_lora.py --model_name_or_path models/Qwen1.5-14B --output_path models/Qwen1.5-14B-lora-merge --lora_path lora/qwen1.5_14b
