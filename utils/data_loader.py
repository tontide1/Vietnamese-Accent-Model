# src/data_loader.py
import os
import requests
import zipfile
import re
from tqdm import tqdm
from .utils import tokenize, remove_vn_accent
import random
import multiprocessing as mp

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

# Hàm phụ trợ cho multiprocessing, cần được định nghĩa ở top-level
def _process_single_sentence_for_splitting(sent_accented: str):
    if sent_accented.strip():
        temp_tokenized_for_unaccenting = tokenize(sent_accented)
        unaccented_words = [remove_vn_accent(word) for word in temp_tokenized_for_unaccenting]
        unaccented_sentence_str = " ".join(unaccented_words)
        tokenized_accented_sentence = tokenize(sent_accented)
        if unaccented_sentence_str and tokenized_accented_sentence:
            return (unaccented_sentence_str, tokenized_accented_sentence)
    return None

def load_and_split_corpus(data_extract_path: str = TRAIN_EXTRACT_PATH, test_size: float = 0.2, random_seed: int = 42):
    """
    Load dữ liệu, tạo phiên bản không dấu, tokenize câu có dấu,
    và chia thành tập huấn luyện và tập kiểm thử.

    Returns:
        tuple: (train_corpus, test_set)
        train_corpus (list[list[str]]): Danh sách các câu có dấu đã tokenize cho huấn luyện.
        test_set (list[tuple[str, list[str]]]): Danh sách các tuple 
                                                (câu không dấu dạng chuỗi, câu có dấu đã tokenize) cho kiểm thử.
    """
    if not os.path.exists(data_extract_path):
        print(f"Error: Training data path not found: {data_extract_path}")
        return [], []

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
        return [], []

    print(f"Loaded {len(full_text_content)} documents.")
    full_data_string = ". ".join(full_text_content)
    full_data_string = full_data_string.replace("\n", ". ")
    
    raw_sentences_with_accent = re.split(r'[.?!]\s+', full_data_string)

    # GIỚI HẠN SỐ LƯỢNG CÂU ĐỂ TRÁNH MemoryError
    MAX_PROCESS_SENTENCES = 1000 # Bạn có thể điều chỉnh con số này
    if len(raw_sentences_with_accent) > MAX_PROCESS_SENTENCES:
        print(f"CẢNH BÁO: Giới hạn xử lý {MAX_PROCESS_SENTENCES} câu đầu tiên trên tổng số {len(raw_sentences_with_accent)} câu để tiết kiệm bộ nhớ.")
        raw_sentences_with_accent = raw_sentences_with_accent[:MAX_PROCESS_SENTENCES]
    
    print(f"Processing {len(raw_sentences_with_accent)} raw sentences for splitting using multiprocessing...")
    
    # Sử dụng multiprocessing Pool
    # Mặc định Pool() sẽ sử dụng số lượng core CPU có sẵn (os.cpu_count())
    with mp.Pool() as pool:
        # Sử dụng imap để có thể dùng tqdm, và filter None results
        # chunksize có thể giúp tăng hiệu suất cho list rất lớn
        # Chọn chunksize dựa trên thực nghiệm, ví dụ len(data) // (num_cores * 4)
        num_cores = os.cpu_count() or 1 # Đảm bảo num_cores ít nhất là 1
        chunk_size = max(1, len(raw_sentences_with_accent) // (num_cores * 4))
        
        # Bọc list đầu vào của pool.imap với tqdm
        # pool.imap sẽ xử lý lười biếng (lazy evaluation)
        results_iterator = pool.imap(_process_single_sentence_for_splitting, 
                                     tqdm(raw_sentences_with_accent, desc="Generating unaccented and tokenizing (parallel)"),
                                     chunksize=chunk_size)
        
        # Lọc bỏ các kết quả None (nếu câu rỗng hoặc không xử lý được)
        processed_sentences = [res for res in results_iterator if res is not None]

    if not processed_sentences:
        print("No sentences could be processed. Check data and tokenization.")
        return [], []
        
    random.seed(random_seed)
    random.shuffle(processed_sentences)
    
    split_index = int(len(processed_sentences) * (1 - test_size))
    train_data_pairs = processed_sentences[:split_index]
    test_data_pairs = processed_sentences[split_index:]
    
    train_corpus = [pair[1] for pair in train_data_pairs] # Chỉ lấy câu có dấu đã tokenize cho training
    test_set = test_data_pairs # Giữ nguyên (câu không dấu, câu có dấu tokenize) cho testing
    
    print(f"Data split: {len(train_corpus)} training sentences, {len(test_set)} test sentences.")
    return train_corpus, test_set

if __name__ == '__main__':
    print("Running data_loader.py directly to check data and load corpus...")
    if check_data_exists():
        print("\nAttempting to load and split corpus...")
        train_corpus, test_set = load_and_split_corpus()

        if train_corpus and test_set:
            print(f"Successfully loaded and split data.")
            print(f"Number of training sentences: {len(train_corpus)}")
            print(f"Number of test sentences/pairs: {len(test_set)}")
            
            print("\nFirst 3 tokenized training sentences:")
            for tc_sent in train_corpus[:3]:
                print(tc_sent)
            
            print("\nFirst 3 test pairs (unaccented_string, tokenized_accented_sentence):")
            for ts_pair in test_set[:3]:
                print(ts_pair)
        elif train_corpus:
             print(f"Successfully loaded training corpus ({len(train_corpus)} sentences), but test set is empty.")
        elif test_set:
            print(f"Successfully loaded test set ({len(test_set)} pairs), but training corpus is empty.")
        else:
            print("Failed to load or split corpus, or corpus is empty after processing.")
    else:
        print("\nRequired data is missing. Cannot load corpus.")