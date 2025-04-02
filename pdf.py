import fitz
import os
import util
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI
import re
import api
import json
from tqdm import tqdm

openai_client = OpenAI(api_key=api.OPENAI_API)

class Chunk:
    def __init__(self, content, relevance=None, summary=None):
        self.content = content
        self.relevance = relevance
        self.summary = summary

    def classify_relevance(self):
        prompt = f"""
        You are a text classifier specialized in policy analysis. Your task is to determine whether the provided text contains information relevant to climate policies. For this task, "climate policies" include any discussion about governmental, international, or organizational decisions, strategies, or actions aimed at addressing climate change. Relevant topics include (but are not limited to) carbon taxes, renewable energy initiatives, climate agreements (e.g., the Paris Agreement), emissions regulations, and adaptation/mitigation strategies.

        Instructions:
        1. Read the text carefully.
        2. If the text includes any discussion of policies, decisions, or actions related to climate change, classify it as **1**.
        3. If the text does not mention any such information, classify it as **0**.
        4. Do not provide any explanations, answer with exactly one number.
        
        Examples:
        - Example 1:
        - Text: "The government introduced a new carbon tax aimed at reducing greenhouse gas emissions."
        - 1
        - Example 2:
        - Text: "The local sports team won their championship game last night."
        - 0

        Now, classify the following text:

        \"\"\"{self.content}\"\"\"
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  
            max_tokens=5,     
        )

        raw_answer = response.choices[0].message.content.strip()
        match = re.search(r'[01]', raw_answer)
        if match:
            self.relevance = 1
        else:
            self.relevance = 0
            
    def summarize_record(self):
        prompt = f"""
        You are an expert in extracting information. Your task is to provide a detailed summary of:
        • The policy or policies mentioned
        • The effect of the policy
        • The country that applied the policy (Use exactly only the country name)
        • The year associated with the policy (Use exactly the year number)

        Use the exact structure below:
        Policy:
        Effect:
        Country:
        Year:

        For any missing or unavailable information, fill it using your knowledge.
        If still not possible, fill Nan.
        Sometimes you get a city name rather than a country name, it is necessary to convert it to a country name.

        Text:
        \"\"\"{self.content}\"\"\"

        Summary:
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  
            max_tokens=300   
        )
        self.summary = response.choices[0].message.content.strip()

    def summarize_knowledge(self):
        prompt = f"""
        You are an expert in extracting information. Your task is to provide a summary of the given context using bullet points.
        The given context is from a manual introducing knowledges of climates or possible effects of climate.

        Text:
        \"\"\"{self.content}\"\"\"

        Summary:
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  
            max_tokens=300   
        )
        self.summary = response.choices[0].message.content.strip()
        
    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    
class PDF:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        folder_names = pdf_path.split('/')[1:-1]
        self.folder_name = "/".join(folder_names)
        file_full_name = pdf_path.split('/')[-1]
        self.pdf_name = file_full_name.split('.')[0]
        self.content = ""
        self.chunks = []

    def extract_text(self, output_folder:str):
        os.makedirs(output_folder, exist_ok=True)
        doc = fitz.open(self.pdf_path)
        all_text = ""

        print(f"Extracting texts from {self.pdf_name}.")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            lines = text.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            cleaned_text = '\n'.join(non_empty_lines)
            all_text += cleaned_text

        self.content = all_text
        os.makedirs(f"{output_folder}/{self.folder_name}", exist_ok=True)
        save_path = f"{output_folder}/{self.folder_name}/{self.pdf_name}.txt"

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(self.content)
        print(f"Extracted text saved to: {save_path}")
            
    def load_content(self, content_file):
        with open(content_file, "r", encoding="utf-8") as file:
            self.content = file.read()
        print(f"{self.pdf_name}'s content is loaded.")

    def naive_chunking(self, chunk_size, overlap):
        text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        splits = text_splitter.split_text(self.content)
        print(f"Total text chunks created: {len(splits)}")
        self.chunks = [Chunk(txt) for txt in splits]

    def save_chunks(self, output_folder):
        data = [chunk.__dict__ for chunk in self.chunks]
        os.makedirs(f"{output_folder}/{self.folder_name}", exist_ok=True)
        with open(f"{output_folder}/{self.folder_name}/{self.pdf_name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Data written to {output_folder}/{self.folder_name}/{self.pdf_name}.json")

    def filter_chunks_by_revelance(self):
        new_chunks = []
        counter = 0
        for chunk in tqdm(self.chunks, desc=f"Filtering {self.pdf_name}'s chunks."):
            if chunk.relevance:
                new_chunks.append(chunk)
            else:
                counter += 1
        self.chunks = new_chunks
        print(f"{counter} chunks are filtered out!")