import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score
from transformers import BitsAndBytesConfig

# QLoRA 설정
lora_config = LoraConfig(
    r=4,  # rank
    lora_alpha=8,  # scaling factor
    lora_dropout=0.1,  # dropout 비율
    target_modules=["q_proj", "v_proj"],  # LoRA를 적용할 모듈
)

# 데이터셋 로드
dataset = load_dataset("json", data_files={
    "train": "/kaggle/input/profiling-data/train_data.json",
    "test": "/kaggle/input/profiling-data/test_data.json"
})

# 모델과 토크나이저 로드
model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# `BitsAndBytesConfig` 객체를 사용하여 8비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 8비트 양자화
    bnb_4bit_quantize=False  # 4비트 양자화 사용하지 않음 (필요시 True로 설정)
)

# 모델을 양자화된 상태로 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    quantization_config=bnb_config  # 양자화 설정을 추가
)

# 'pad_token'이 없는 경우, eos_token을 대신 사용
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA 모델 적용
model = get_peft_model(model, lora_config)

# 데이터 전처리 함수
def preprocess_function(examples):
    inputs = [f"{instruction}{input_}" for instruction, input_ in zip(examples["instruction"], examples["input"])]
    targets = [f"{out}" for out in examples["output"]]

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)

    # pad token을 -100으로 처리하여 학습 시 무시
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_batch]
        for label_batch in labels["input_ids"]
    ]
    return model_inputs

# 데이터셋에 전처리 함수 적용
train_data = dataset['train'].map(preprocess_function, batched=True)
test_data = dataset['test'].map(preprocess_function, batched=True)

# 불필요한 열 제거
train_data = train_data.remove_columns(['instruction', 'input', 'output'])
test_data = test_data.remove_columns(['instruction', 'input', 'output'])

# 평가 지표 계산 함수
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(predictions, dim=-1)
    # 정확도 계산
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=10,
    save_steps=20,
    learning_rate=5e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    fp16=True,
    remove_unused_columns=False,
    metric_for_best_model="accuracy",  # 가장 좋은 모델을 선택하는 지표 설정
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics  # 평가 지표 함수 추가
)

# 모델 학습
trainer.train()

# 학습 후 모델 저장
model.save_pretrained('./fine_tuned_model_lora')
tokenizer.save_pretrained('./fine_tuned_model_lora')
