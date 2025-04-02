import util
import os
import json
from collections import defaultdict

class Policy:
    def __init__(self, policy_id, policy=None, effect=None, country=None, year=None):
        self.policy_id = policy_id
        self.policy = policy
        self.effect = effect
        self.country = country
        self.year = year

    def load_policy(self, item):
        summary = item['summary']
        seg1 = summary.split('Policy:')[-1]
        seg2 = seg1.split('Effect:')[-1]
        seg3 = seg2.split('Country:')[-1]
        self.policy = seg1.split('Effect:')[0].replace("\n","").strip()
        self.effect = seg2.split('Country:')[0].replace("\n","").strip()
        country = seg3.split('Year:')[0].replace("\n","").strip()
        self.country = country.replace('/','')
        self.year = seg3.split('Year:')[-1].replace("\n","").strip()
    
    def save_policy(self, file_path):
        data = self.__dict__
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except (json.JSONDecodeError, IOError):
                existing_data = []
        else:
            existing_data = []
        existing_data.append(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)
        print(f"Data written to {file_path}")

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

def group_by(attribute, input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for item in data:
        value = item.get(attribute).replace('/','')
        grouped[value].append(item)

    for value, items in grouped.items():
        if len(value) > 20:
            value = value[:20]        
        output_file = os.path.join(output_dir, f"{value}.json")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(items, outfile, indent=2, ensure_ascii=False)

    print(f"Grouped data has been saved in '{output_dir}' directory.")
