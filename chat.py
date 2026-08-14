import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "google/gemma-2-2b-it"
adapter_path = "./my_digital_double_model"

print("--- Загрузка цифрового двойника... ---")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Подключаем твои обученные слои
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

print("\n--- ГОТОВО! Можешь общаться. (напиши 'exit' для выхода) ---")

while True:
    user_input = input("Ты: ")
    if user_input.lower() == 'exit':
        break

    # Форматируем запрос как при обучении
    prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )

    # Декодируем только ответ модели
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.split("model")[-1].strip()

    print(f"Двойник: {response}\n")
