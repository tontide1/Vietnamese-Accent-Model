# utils/predictor.py
import os
import pickle
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.utils import gen_accents_word # For beam_search thuật toán
from utils.data_loader import SYLLABLES_PATH, check_data_exists 
from utils.model_trainer import MODEL_DIR, DEFAULT_MODEL_FILENAME

# --- Configuration ---
# MODEL_DIR and DEFAULT_MODEL_FILENAME are imported from model_trainer
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, DEFAULT_MODEL_FILENAME)

# --- Prediction Functions ---
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Please train and save the model first (e.g., by running model_trainer.py) or provide a valid path.")
        return None
    
    print(f"Loading model from {model_path}...")
    try:
        with open(model_path, 'rb') as fin:
            model_loaded = pickle.load(fin)
        print(f"Model loaded successfully. Vocabulary size: {len(model_loaded.vocab)}")
        return model_loaded
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle model from {model_path}. File might be corrupted or not a pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
    return None

_detokenizer = TreebankWordDetokenizer()

def beam_search_predict_accents(text_no_accents: str, model, k: int = 3, 
                                syllables_file: str = SYLLABLES_PATH, 
                                detokenizer=_detokenizer) -> list[tuple[str, float]]:
    """
    Predicts accents for a sentence using beam search.

    :param text_no_accents: Input sentence string without accents.
    :param model: Trained NLTK language model.
    :param k: Beam width.
    :param syllables_file: Path to the Vietnamese syllables file for gen_accents_word.
    :param detokenizer: NLTK detokenizer instance.
    :return: A list of (sentence_with_accents, log_score) tuples, sorted by score.
    """
    words = text_no_accents.lower().split()
    sequences = [] # Stores list of ([word_sequence], score)

    for idx, word_no_accent in enumerate(words):
        possible_accented_words = gen_accents_word(word_no_accent, syllables_path=syllables_file)
        if not possible_accented_words:
            possible_accented_words = {word_no_accent} 

        if idx == 0:
            sequences = [([x], 0.0) for x in possible_accented_words]
        else:
            all_new_sequences = []
            for seq_words, seq_score in sequences:
                for next_accented_word in possible_accented_words:
                    context = seq_words[-(model.order - 1):] if model.order > 1 else [] 
                    try:
                        score_addition = model.logscore(next_accented_word, tuple(context))
                    except Exception as e: # Could be due to OOV in context or word itself
                        print(f"Logscore error for '{next_accented_word}' with context {context}: {e}. Assigning low score.")
                        score_addition = -float('inf') 
                        
                    new_seq_words = seq_words + [next_accented_word]
                    all_new_sequences.append((new_seq_words, seq_score + score_addition))
            
            all_new_sequences = sorted(all_new_sequences, key=lambda x: x[1], reverse=True)
            sequences = all_new_sequences[:k]
            if not sequences: 
                if all_new_sequences:
                    sequences = [(all_new_sequences[0][0][:-1] + [word_no_accent], all_new_sequences[0][1] - 1000)] 
                else:
                    return []

    results = [(detokenizer.detokenize(seq_words), score) for seq_words, score in sequences]
    return results

if __name__ == '__main__':
    print("Running predictor.py directly...")

    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' resource not found. Please download it first by running: import nltk; nltk.download('punkt')")
        exit()
    except ImportError:
        print("NLTK library not found. Please install it: pip install nltk")
        exit()

    print("\nStep 1: Checking data (like vn_syllables.txt) availability...")
    if not check_data_exists():
        print(f"Critical: Required data (including {SYLLABLES_PATH}) not found. Accent prediction might fail or be impaired.")
        print("Please ensure data is correctly placed as per README or run 'python main.py --action check_data'.")
    else:
        print("Data check indicates necessary files are present.")

    # 2. Load the trained model
    print("\nStep 2: Loading the trained model...")
    model = load_model(DEFAULT_MODEL_PATH)

    if model:
        print("\nStep 3: Testing accent prediction...")
        sentence_no_accents = "ngay hom qua la ngay bau cu tong thong my"
        print(f"Input (no accents): {sentence_no_accents}")
        
        predictions = beam_search_predict_accents(sentence_no_accents, model, k=3, syllables_file=SYLLABLES_PATH)
        
        if predictions:
            print("Top predictions:")
            for i, (sent, score) in enumerate(predictions):
                print(f"{i+1}. '{sent}' (Score: {score:.4f})")
        else:
            print("Accent prediction failed to return results.")
            
        sentence_no_accents_2 = "chuc mung nam moi"
        print(f"\nInput (no accents): {sentence_no_accents_2}")
        predictions_2 = beam_search_predict_accents(sentence_no_accents_2, model, k=3, syllables_file=SYLLABLES_PATH)
        if predictions_2:
            print("Top predictions:")
            for i, (sent, score) in enumerate(predictions_2):
                print(f"{i+1}. '{sent}' (Score: {score:.4f})")
        else:
            print("Accent prediction failed to return results for the second sentence.")

    else:
        print("Model could not be loaded. Cannot run predictions.")
    
    print("\nPredictor script finished.")