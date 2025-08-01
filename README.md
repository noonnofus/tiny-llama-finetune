# Tiny-Llama 모델 LoRA 방식 기반으로 fine-tuning

## 📜 Description
이 리포지토리는 LoRA(Low-Rank Adaptation) 방식을 활용하여 TinyLlama-1.1B-Chat 모델을 fine-tuning한 프로젝트입니다.
모든 학습 과정은 Google Colab 환경에서 수행되었으며, 학습에 사용된 데이터셋은 [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca/tree/main/assets) 데이터셋입니다.

## 🧠 Fine-tuning
### 1. Hugging Face 로그인 및 모델 불러오기
Hugging Face에 로그인한 후 TinyLlama 모델을 불러오는 코드입니다.
로그인 시에는 hf_로 시작하는 토큰이 필요하며, 해당 token에 최소한 "Read" 권한이 있어야 모델을 불러올 수 있습니다.
```python3
login("token")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # 원하는 모델

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
)
```
### 2. Alpaca 데이터셋 불러오기 및 학습용 프롬프트 형식으로 변환

Stanford Alpaca JSON 데이터를 Hugging Face의 `datasets.Dataset` 형식으로 변환합니다.  
이때 모델 학습에 적합하도록 `Instruction`, `Input`, `Response` 구조로 가공합니다.

**Before:**
```json
{
  "instruction": "Create a classification algorithm for a given dataset.", 
  "input": "Dataset of medical images", 
  "output": "We can use a convolutional neural network (CNN) for classifying medical images. ..."
}
```
**After:**
```
### Instruction:
Create a classification algorithm for a given dataset.

### Input:
Dataset of medical images

### Response:
We can use a convolutional neural network (CNN) for classifying medical images. ...
```
***Code:***
```python3
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
```
### 3. LoRA 설정 및 모델 적용
### 4. 데이터셋 토큰화
### 5. TrainingArguments 설정
### 6. Trainer 구성 및 학습 시작

## 📦 Installation 
```bash
# 해당 리포지토리 클론후 이동
git clone https://github.com/noonnofus/tiny-llama-finetune.git && cd tiny-llama-finetune

# 의존성 설치
pip install -r requirements.txt

# 학습 시작
python3 script.py
```
