import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Проверка импортов...")
try:
    # Если эта строка не упадет, значит torch чист
    import transformers.models.gemma
    print("Успех! Ошибка 'intl' не обнаружена.")
except Exception as e:
    print(f"Ошибка все еще тут: {e}")
