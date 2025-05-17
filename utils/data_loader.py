# src/data_loader.py
import os
import requests
import zipfile
import re
from tqdm import tqdm
from .utils import tokenize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_ZIP_URL = "https://github.com/hoanganhpham1006/Vietnamese_Language_Model/raw/master/Train_Full.zip"
TRAIN_ZIP_NAME = "Train_Full.zip"
TRAIN_ZIP_PATH = os.path.join(DATA_DIR, TRAIN_ZIP_NAME)
TRAIN_EXTRACT_PATH = os.path.join(DATA_DIR, "Train_Full")

SYLLABLES_URL = "https://gist.githubusercontent.com/hieuthi/0f5adb7d3f79e7fb67e0e499004bf558/raw/135a4d9716e49a981624474156d6f247b9b46f6a/all-vietnamese-syllables.txt"
SYLLABLES_NAME = "vn_syllables.txt"
SYLLABLES_PATH = os.path.join(DATA_DIR, SYLLABLES_NAME)


def check_data_exists():
    """kiểm tra xem dữ liệu đã có chưa"""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory configured at: {DATA_DIR}")
    all_data_exists = True
    if not os.path.exists(TRAIN_EXTRACT_PATH) or not os.listdir(TRAIN_EXTRACT_PATH):
        print(f"Warning: Training data directory {TRAIN_EXTRACT_PATH} is missing or empty.")
        print("Please ensure you have downloaded and extracted 'Train_Full.zip' into the 'data/Train_Full' directory.")
        all_data_exists = False
    else:
        print(f"Training data found at: {TRAIN_EXTRACT_PATH}")

    if not os.path.exists(SYLLABLES_PATH):
        print(f"Warning: Vietnamese syllables file {SYLLABLES_PATH} is missing.")
        print("Please ensure you have 'vn_syllables.txt' in the 'data' directory.")
        all_data_exists = False
    else:
        print(f"Vietnamese syllables file found at: {SYLLABLES_PATH}")
    return all_data_exists


def load_corpus(data_extract_path: str = TRAIN_EXTRACT_PATH) -> list[list[str]]:
    """
    load dữ liệu từ thư mục đã giải nén
    """
    if not os.path.exists(data_extract_path):
        print(f"Error: Training data path not found: {data_extract_path}")
        print("Please run download_and_prepare_data() first or ensure data is correctly placed.")
        return []

    full_text_content = []
    print(f"Loading corpus from: {data_extract_path}")

    for dirname, _, filenames in os.walk(data_extract_path):
        for filename in tqdm(filenames, desc=f"Reading files in {os.path.basename(dirname)}"):
            if filename.endswith(".txt"): 
                try:
                    with open(os.path.join(dirname, filename), 'r', encoding='UTF-16') as f:
                        full_text_content.append(f.read())
                except Exception as e:
                    print(f"Could not read file {os.path.join(dirname, filename)}: {e}")
    
    if not full_text_content:
        print("No text files found or loaded from the training data path.")
        return []

    print(f"Loaded {len(full_text_content)} documents.")
    full_data_string = ". ".join(full_text_content)
    full_data_string = full_data_string.replace("\n", ". ")
    
    corpus = []
    raw_sents = re.split(r'[.?!]\s+', full_data_string) 
    print(f"Processing {len(raw_sents)} raw sentences...")
    for sent in tqdm(raw_sents, desc="Tokenizing sentences"):
        if sent.strip(): # Ensure sentence is not just whitespace
            corpus.append(tokenize(sent))
    
    print(f"Corpus created with {len(corpus)} tokenized sentences.")
    return corpus

if __name__ == '__main__':
    print("Running data_loader.py directly to check data and load corpus...")
    if check_data_exists():
        print("\nAttempting to load corpus...")
        loaded_corpus = load_corpus()
        if loaded_corpus:
            print(f"Successfully loaded corpus. Number of sentences: {len(loaded_corpus)}")
            print("First 3 tokenized sentences:")
            for c_sent in loaded_corpus[:3]:
                print(c_sent)
        else:
            print("Failed to load corpus or corpus is empty.")
    else:
        print("\nRequired data is missing. Cannot load corpus.")