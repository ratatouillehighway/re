import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# LoRA 설정
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=16,  # scaling factor
    lora_dropout=0.1,  # dropout 비율
    target_modules=["q_proj", "v_proj"],  # LoRA를 적용할 모듈 (attention의 쿼리 및 값 프로젝터)
)

# LoRA가 적용된 모델을 생성


dataset = load_dataset("json", data_files={"train": "./data/train_data.json", "test": "./data/test_data.json"})

model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=None,
)

# 'pad_token'이 없는 경우, eos_token을 대신 사용
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = get_peft_model(model, lora_config)
def preprocess_function(examples):
    inputs = [f"{instruction}{input_}" for instruction, input_ in zip(examples["instruction"], examples["input"])]
    targets = [f"{out}" for out in examples["output"]]

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=1024)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=1024)

    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_batch]
        for label_batch in labels["input_ids"]
    ]

    return model_inputs

train_data = dataset['train'].map(preprocess_function, batched=True)
test_data = dataset['test'].map(preprocess_function, batched=True)

print(train_data[0])

training_args = TrainingArguments(
    output_dir="./results",  # 결과를 저장할 폴더
    eval_strategy="epoch",  # 평가 주기
    save_strategy="epoch",
    learning_rate=5e-3,  # 학습률
    per_device_train_batch_size=8,  # 배치 사이즈
    per_device_eval_batch_size=8,  # 평가 배치 사이즈
    num_train_epochs=3,  # 학습 epoch 수
    weight_decay=0.01,  # 가중치 감소
    logging_dir="./logs",  # 로그 폴더
    logging_steps=10,  # 로그 빈도
    save_steps=500,  # 체크포인트 저장 주기
    load_best_model_at_end=True,  # 최상의 모델을 끝에 로드
    fp16=False,  # fp16 대신 bfloat16을 사용하므로 fp16은 False
    bf16=True,  # bfloat16 사용
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,  # LoRA가 적용된 모델
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# 모델 학습
trainer.train()

# 학습 후 모델 저장
model.save_pretrained('./fine_tuned_model_lora')
tokenizer.save_pretrained('./fine_tuned_model_lora')
