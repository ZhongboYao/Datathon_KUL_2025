import os
import shutil
import json

def clear_output_folder(folder_path:str):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"{folder_path} is cleared!")
    os.makedirs(folder_path, exist_ok=True)
    print(f"{folder_path} is ready for new content.")

def get_individual_file_path(folder:str, form:str) -> list[str]: 
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(form)]

def create_class_from_json(cls, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    instances = [cls.from_dict(data) for data in data_list]
    return instances

def save_as_json(content, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    print(f"Saved content to {path}")

def load_json(path):
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # print(f"Loaded content from {path}")
    return data