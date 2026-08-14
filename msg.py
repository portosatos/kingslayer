import json

import ijson


def fast_parse_tg(json_path, my_name, friend_name):
    print("Запускаю потоковый парсинг через ijson...")
    dataset = []
    blocks = []
    current_block = {"from": None, "text": ""}

    with open(json_path, "r", encoding="utf-8") as f:
        # Итерируемся прямо по объектам в списке сообщений
        # Путь в Telegram JSON: chats -> list -> item -> messages -> item
        parser = ijson.items(f, "chats.list.item.messages.item")

        count = 0
        try:
            for msg in parser:
                author = msg.get("from")
                text_data = msg.get("text", "")

                # Текстовое поле может быть списком в TG
                text = ""
                if isinstance(text_data, list):
                    for part in text_data:
                        if isinstance(part, str):
                            text += part
                        elif isinstance(part, dict):
                            text += part.get("text", "")
                else:
                    text = text_data

                if not text or not author:
                    continue

                # Нам нужны только сообщения от тебя или Дэнка
                if author not in [my_name, friend_name]:
                    continue

                if author == current_block["from"]:
                    current_block["text"] += " " + text
                else:
                    if current_block["from"]:
                        blocks.append(current_block.copy())
                    current_block = {"from": author, "text": text}

                count += 1
                if count % 10000 == 0:
                    print(f"Обработано {count} сообщений...")

        except Exception as e:
            print(f"Парсинг прерван ошибкой (возможно, конец файла): {e}")

    # Формируем пары "Друг -> Я"
    print("Склеиваю диалоги в пары...")
    for i in range(len(blocks) - 1):
        if blocks[i]["from"] == friend_name and blocks[i + 1]["from"] == my_name:
            dataset.append(
                {
                    "instruction": blocks[i]["text"].strip(),
                    "output": blocks[i + 1]["text"].strip(),
                }
            )

    return dataset


# Запуск (имена берем из твоих скриншотов)
my_id_name = "not aurora anymore"
friend_id_name = "Дэнк"

result = fast_parse_tg("result.json", my_id_name, friend_id_name)

if result:
    with open("dataset.jsonl", "w", encoding="utf-8") as f:
        for entry in result:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Успех! Создано {len(result)} пар. Теперь твоя RTX 2080 Ti готова к работе!")
