import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# Переходим на 3-ю версию
model_id = "google/gemma-3-4b-it"

# 1. Загрузка данных (твой формат)
def prepare_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    formatted = [
        {"text": f"<start_of_turn>user\n{item['instruction']}<end_of_turn>\n<start_of_turn>model\n{item['output']}<end_of_turn>"}
        for item in lines
    ]
    return Dataset.from_list(formatted)

print("--- Шаг 1: Подготовка данных ---")
dataset = prepare_dataset("dataset.jsonl")

# 2. Квантование для RTX 2080 Ti (11GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 3. Загрузка модели (ОБХОД ОШИБОК)
print("--- Шаг 2: Загрузка Gemma 3 4B ---")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    # ФИКС: отключаем новые маски, требующие PyTorch 2.6
    attn_implementation="eager"
)
model.config.use_cache = False

# 4. LoRA под 4B версию
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5. Токенизация (ВАЖНО для Gemma 3)
def tokenize_function(examples):
    # ФИКС: Gemma 3 требует эти ID для обучения
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_token_type_ids=True
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("--- Шаг 3: Токенизация ---")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 6. Настройки тренера
training_args = TrainingArguments(
    output_dir="./gemma_v3_double",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, # Больше шагов, так как 4B тяжелее
    warmup_steps=10,
    max_steps=150,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="no",
    # ФИКС: не даем удалять колонку token_type_ids
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("\n--- ЗАПУСК ОБУЧЕНИЯ GEMMA 3 ---")
trainer.train()

# 7. Сохранение (в другую папку!)
model.save_pretrained("./my_digital_double_v3")
tokenizer.save_pretrained("./my_digital_double_v3")
print("\nГотово! Модель в папке: my_digital_double_v3")
