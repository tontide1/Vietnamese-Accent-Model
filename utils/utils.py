# src/utils.py
import re
import string
from nltk import word_tokenize
import os

VN_SYLLABLES_FILE_PATH = "data/vn_syllables.txt"

def tokenize(doc: str) -> list[str]:
    """
    Tokenizes a document, converts to lowercase, removes punctuation (except '_'),
    and filters out empty tokens.
    """
    tokens = word_tokenize(doc.lower())
    # Allow underscore, remove other punctuation
    table = str.maketrans('', '', string.punctuation.replace("_", ""))
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]  # Remove empty strings
    return tokens

def remove_vn_accent(word: str) -> str:
    """
    Removes Vietnamese accents from a word.
    Converts the word to lowercase before processing.
    """
    word = word.lower()
    word = re.sub(r'[áàảãạăắằẳẵặâấầẩẫậ]', 'a', word)
    word = re.sub(r'[éèẻẽẹêếềểễệ]', 'e', word)
    word = re.sub(r'[óòỏõọôốồổỗộơớờởỡợ]', 'o', word)
    word = re.sub(r'[íìỉĩị]', 'i', word)
    word = re.sub(r'[úùủũụưứừửữự]', 'u', word)
    word = re.sub(r'[ýỳỷỹỵ]', 'y', word)
    word = re.sub(r'đ', 'd', word)
    return word

def gen_accents_word(word: str, syllables_path: str = VN_SYLLABLES_FILE_PATH) -> set[str]:
    """
    Generates a set of possible Vietnamese words with accents for a given word
    without accents (or with accents, it will normalize first).
    Requires the 'vn_syllables.txt' file specified by syllables_path.
    """
    normalized_input_word = word.lower()
    word_no_accent = remove_vn_accent(normalized_input_word)
    all_accent_word = {normalized_input_word}  # Start with the input word (normalized)

    if not os.path.exists(syllables_path):
        print(f"Warning: Syllables file not found at {syllables_path}. "
              f"Accent generation will be limited to the input word: '{word}'.")
        # Optionally, add the purely unaccented version if different
        if word_no_accent != normalized_input_word:
            all_accent_word.add(word_no_accent)
        return all_accent_word

    try:
        with open(syllables_path, 'r', encoding='utf-8') as f:
            for w_line in f.read().splitlines():
                w_line_lower = w_line.lower()  # Normalize file content
                if remove_vn_accent(w_line_lower) == word_no_accent:
                    all_accent_word.add(w_line_lower)
    except Exception as e:
        print(f"Error reading or processing syllables file {syllables_path}: {e}")
    
    return all_accent_word

if __name__ == '__main__':
    # Example Usage (for testing utils.py directly)
    print("Tokenize example:")
    print(tokenize("Đây_là một câu, ví dụ."))

    print("\nRemove accent example:")
    print(remove_vn_accent("hoàng"))
    print(remove_vn_accent("Hoàng"))

    print("\nGenerate accents example (make sure 'data/vn_syllables.txt' exists or is created):")
    
    # Create a dummy data directory and vn_syllables.txt for testing if they don't exist
    test_data_dir = "data"
    test_syllables_file = os.path.join(test_data_dir, "vn_syllables.txt") # Consistent with VN_SYLLABLES_FILE_PATH

    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
        print(f"Created directory: {test_data_dir}")

    if not os.path.exists(test_syllables_file):
        with open(test_syllables_file, "w", encoding="utf-8") as f:
            f.write("hoàng\nhoang\nhọang\nhởang\nhờang\nhoa\nhòa\nviệt\n") # Add some test data
        print(f"Created dummy {test_syllables_file} for testing.")
    else:
        print(f"Using existing {test_syllables_file} for testing.")

    print(f"'hoang': {gen_accents_word('hoang', syllables_path=test_syllables_file)}")
    print(f"'hoa': {gen_accents_word('hoa', syllables_path=test_syllables_file)}")
    print(f"'viet': {gen_accents_word('viet', syllables_path=test_syllables_file)}")
    print(f"'HoÀnG': {gen_accents_word('HoÀnG', syllables_path=test_syllables_file)}") # Test case normalization