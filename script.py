from dotenv import load_dotenv
import os
from huggingface_hub import login
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch

load_dotenv()

token = os.getenv("HF_TOKEN")

login(token)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
)

with open("dataset/alpaca_data.json", "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

def generate_prompt(ex):
  if ex["input"]:
    return f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
  else:
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"

dataset = dataset.map(lambda x: {"prompt": generate_prompt(x)})

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

def tokenize_function(example):
    result = tokenizer(
        example["prompt"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./tinyllama-lora",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator, 
)

trainer.train()
print('모델 훈련이 완료되었습니다.')
