# src/model_trainer.py
import os
import pickle
# from nltk.lm.preprocessing import padded_everygram_pipeline # Removed
# from nltk.lm import KneserNeyInterpolated, Laplace, WittenBellInterpolated # Removed
from collections import defaultdict # Added
from utils import data_loader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_FILENAME = "kneserney_trigram_model.pkl"
N_GRAM_ORDER = 3

# --- Model Training Functions ---
def train_ngram_model(corpus: list[list[str]],
                      n: int = N_GRAM_ORDER): # Signature changed, model_class removed
    if not corpus:
        print("Corpus is empty. Cannot train model.")
        return None

    print(f"Preparing data for custom {n}-gram model...")

    start_symbol = "<s>"
    end_symbol = "</s>"

    ngram_counts = defaultdict(int)
    context_counts = defaultdict(int) # For (n-1)-grams
    vocab = set()

    for sentence_tokens in corpus:
        # Pad sentence: (n-1) start symbols, 1 end symbol.
        # e.g., for n=3 (trigram): ['<s>', '<s>'] + sentence_tokens + ['</s>']
        # e.g., for n=1 (unigram): [] + sentence_tokens + ['</s>']
        current_padded_sentence = ([start_symbol] * (n - 1)) + sentence_tokens + [end_symbol]

        for token in sentence_tokens: # Original sentence tokens for vocab
            vocab.add(token)

        # Generate n-grams and (n-1)-gram contexts
        # Iterate up to the point where the last n-gram can be formed
        for i in range(len(current_padded_sentence) - n + 1):
            ngram = tuple(current_padded_sentence[i : i + n])
            ngram_counts[ngram] += 1

            if n > 1:
                # The context is the first (n-1) tokens of the n-gram
                context = tuple(current_padded_sentence[i : i + n - 1])
                context_counts[context] += 1
            # For n=1 (unigrams), context_counts will be handled later if needed for P(w) = count(w)/TotalWords

    # For unigram model (n=1), if we define P(w) = count(w) / total_tokens,
    # context_counts can store the total number of tokens.
    if n == 1:
        total_word_occurrences = sum(ngram_counts.values()) # Sum of counts of all unigrams (word occurrences)
        if total_word_occurrences > 0:
            context_counts[()] = total_word_occurrences # Global context for unigrams is the total count

    print(f"Custom model training complete. Vocabulary size: {len(vocab)}")
    print(f"Number of unique {n}-grams: {len(ngram_counts)}")
    if n > 1 or (n == 1 and context_counts):
        print(f"Number of unique contexts: {len(context_counts)}")

    # The "model" is now a dictionary of these counts, vocab, and n
    custom_model = {
        "n": n,
        "vocab": list(vocab), # Store as list for potential JSON needs, pickle handles sets
        "ngram_counts": dict(ngram_counts), # Convert defaultdict to dict
        "context_counts": dict(context_counts), # Convert defaultdict to dict
    }
    return custom_model

def save_model(model, 
               model_dir: str = MODEL_DIR, 
               filename: str = DEFAULT_MODEL_FILENAME):
    if model is None:
        print("Model is None. Nothing to save.")
        return False

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    print(f"Saving model to {model_path}...")
    try:
        with open(model_path, 'wb') as fout:
            pickle.dump(model, fout)
        print(f"Model successfully saved to {model_path}")
        return True
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")
        return False

if __name__ == '__main__':
    print("Running model_trainer.py directly...")

    # #1.  kiểm tra xem nltk đã được cài punkt chưa (Đã loại bỏ vì không còn dùng NLTK trực tiếp trong train_ngram_model)
    # try:
    #     import nltk
    #     nltk.data.find('tokenizers/punkt')
    # except nltk.downloader.DownloadError:
    #     print("NLTK 'punkt' resource not found. Please download it first by running:")
    #     print("import nltk; nltk.download('punkt')")
    #     exit()
    # except ImportError:
    #     print("NLTK library not found. Please install it: pip install nltk")
    #     exit()

    # 2. down dữ liệu 
    print("\nStep 1: Downloading and preparing data...")
    # data_loader.download_and_prepare_data() # GHI CHÚ: Hàm này không được định nghĩa trong data_loader.py.
                                          # Đảm bảo dữ liệu có sẵn tại data_loader.TRAIN_EXTRACT_PATH

    # 3. Load corpus
    print("\nStep 2: Loading corpus...")
    # TRAIN_EXTRACT_PATH is defined in data_loader, ensure it's correct
    corpus = data_loader.load_corpus(data_extract_path=data_loader.TRAIN_EXTRACT_PATH)

    if not corpus:
        print("Corpus could not be loaded. Exiting trainer.")
    else:
        print(f"Corpus loaded with {len(corpus)} sentences.")
        # 4. Train model
        print("\nStep 3: Training model...")
        # có thể test Laplace, WittenBellInterpolated
        # trained_model = train_ngram_model(corpus, n=N_GRAM_ORDER, model_class=KneserNeyInterpolated) # Old call
        trained_model = train_ngram_model(corpus, n=N_GRAM_ORDER) # New call, model_class removed

        if trained_model:
            # 5. Save model
            print("\nStep 4: Saving model...")
            save_model(trained_model, model_dir=MODEL_DIR, filename=DEFAULT_MODEL_FILENAME)
        else:
            print("Model training failed.")
    print("\nModel training script finished.")