# src/model_trainer.py
import os
import pickle
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated, Laplace, WittenBellInterpolated
from utils import data_loader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_FILENAME = "kneserney_trigram_model.pkl"
N_GRAM_ORDER = 3

# --- Model Training Functions ---
def train_ngram_model(corpus: list[list[str]], 
                      n: int = N_GRAM_ORDER, 
                      model_class = KneserNeyInterpolated):
    if not corpus:
        print("Corpus is empty. Cannot train model.")
        return None

    print(f"Preparing data for {n}-gram model...")
    train_data, padded_sents = padded_everygram_pipeline(n, corpus)
    
    print(f"Training {model_class.__name__} model with n={n}...")
    ngram_model = model_class(n)
    ngram_model.fit(train_data, padded_sents)
    
    print(f"Model training complete. Vocabulary size: {len(ngram_model.vocab)}")
    return ngram_model

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

    # Ensure NLTK 'punkt' is downloaded (usually done once manually or in setup)
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' resource not found. Please download it first by running:")
        print("import nltk; nltk.download('punkt')")
        exit()
    except ImportError:
        print("NLTK library not found. Please install it: pip install nltk")
        exit()

    # 2. Download and prepare data
    print("\nStep 1: Downloading and preparing data...")
    data_loader.download_and_prepare_data()

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
        trained_model = train_ngram_model(corpus, n=N_GRAM_ORDER, model_class=KneserNeyInterpolated)

        if trained_model:
            # 5. Save model
            print("\nStep 4: Saving model...")
            save_model(trained_model, model_dir=MODEL_DIR, filename=DEFAULT_MODEL_FILENAME)
        else:
            print("Model training failed.")
    print("\nModel training script finished.")