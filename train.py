import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# МЕНЯЕМ НА GEMMA 2 (она стабильна на Torch 2.5)
model_id = "google/gemma-2-2b-it"

# 1. Подготовка данных (под твой формат из image_32d4bc.jpg)
def prepare_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    formatted = [
        {"text": f"<start_of_turn>user\n{item['instruction']}<end_of_turn>\n<start_of_turn>model\n{item['output']}<end_of_turn>"}
        for item in lines
    ]
    return Dataset.from_list(formatted)

print("--- Шаг 1: Загрузка данных ---")
dataset = prepare_dataset("dataset.jsonl")

# 2. Квантование (для RTX 2080 Ti)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 3. Загрузка модели
print("--- Шаг 2: Загрузка Gemma 2 ---")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.config.use_cache = False

# 4. Настройка LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5. Токенизация (Gemma 2 не требует token_type_ids, это упрощает всё)
def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("--- Шаг 3: Токенизация ---")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 6. Тренер
training_args = TrainingArguments(
    output_dir="./gemma_v2_double",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, # Увеличил для стабильности на 11ГБ
    warmup_steps=10,
    max_steps=200,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("\n--- СТАРТ ОБУЧЕНИЯ (Gemma 2) ---")
trainer.train()

# 7. Сохранение
model.save_pretrained("./my_digital_double_model")
tokenizer.save_pretrained("./my_digital_double_model")
print("\nПОБЕДА! Модель сохранена в my_digital_double_model")
