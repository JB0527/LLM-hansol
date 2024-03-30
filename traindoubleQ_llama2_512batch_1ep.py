#!/usr/bin/env python
# coding: utf-8

import os,torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from transformers import TrainingArguments, set_seed
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
set_seed(42)
import wandb
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[4]:


secret_wandb = "7df92b5f77c812550e3f38029dc2c0a7bb2b7caa"
wandb.login(key = secret_wandb)
run = wandb.init(
    project='Fine tuning Edentns-DataVortexS with train_combined_doubleQ', 
    job_type="training", 
    anonymous="allow"
)

model_name = "Coldbrew9/Edentns-DataVortexS-for-RAG-10ep-32batch"


from datasets import load_from_disk

dataset = load_from_disk("train_combined_doubleQ")

# 데이터셋 확인
print(f"Dataset load complete : {len(dataset)}")


# ## Model Fine-tuning


import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

bnb_config = BitsAndBytesConfig(  
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,  
        device_map="auto",
        trust_remote_code=True,
        
    )

model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token


# In[21]:


#Adding the adapters in the layers
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
model = get_peft_model(model, peft_config)

print(f"model load complete ")

training_arguments = TrainingArguments(
    output_dir="Fine tuning Edentns-DataVortexS with train_combined_doubleQ",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=1,
    learning_rate=3e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    #warmup_steps=100,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    run_name = "Fine tuning Edentns-DataVortexS with train_combined_doubleQ",
    report_to="wandb"
)


# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)


print(f"starting train ")

trainer.train(resume_from_checkpoint = False)

new_model = "Fine tuning Edentns-DataVortexS with train_combined_doubleQ"

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

## repo
HUGGINGFACE_AUTH_TOKEN = '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' # https://huggingface.co/settings/token
MODEL_SAVE_HUB_PATH = 'Coldbrew9/Edentns-DataVortexS-trainWithCombi'
## Push to huggingface-hub
trainer.model.push_to_hub(
			MODEL_SAVE_HUB_PATH, 
			use_temp_dir=True, 
			use_auth_token=HUGGINGFACE_AUTH_TOKEN
)
trainer.tokenizer.push_to_hub(
			MODEL_SAVE_HUB_PATH, 
			use_temp_dir=True, 
			use_auth_token=HUGGINGFACE_AUTH_TOKEN
)