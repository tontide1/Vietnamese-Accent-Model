# train_model.py
import argparse
import os
import nltk

from utils import data_loader
from utils.data_loader import check_data_exists
from utils import model_trainer
from nltk.lm import KneserNeyInterpolated, Laplace, WittenBellInterpolated

MODEL_CLASSES = {
    "KneserNeyInterpolated": KneserNeyInterpolated,
    "Laplace": Laplace,
    "WittenBellInterpolated": WittenBellInterpolated
}

def ensure_nltk_punkt():
    """Ensures NLTK 'punkt' tokenizer models are available."""
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' resource found.")
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' resource not found. Downloading...")
        nltk.download('punkt')
        print("'punkt' downloaded successfully.")
    except ImportError:
        print("NLTK library not found. Please install it: pip install nltk")
        exit(1)
    except Exception as e:
        print(f"An error occurred with NLTK 'punkt': {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Vietnamese Trigram Model Trainer")
    parser.add_argument("--action", type=str, default="train", 
                        choices=["check_data", "train"],
                        help="Action to perform: 'check_data' verifies data, 'train' trains the model.")
    parser.add_argument("--n_gram", type=int, default=model_trainer.N_GRAM_ORDER, 
                        help="N-gram order for training.")
    parser.add_argument("--model_type", type=str, default="KneserNeyInterpolated", 
                        choices=MODEL_CLASSES.keys(), help="NLTK model class to use for training.")
    parser.add_argument("--model_file", type=str, default=model_trainer.DEFAULT_MODEL_FILENAME,
                        help="Model filename for saving.")

    args = parser.parse_args()

    print("--- Vietnamese Language Model Trainer ---")
    ensure_nltk_punkt()

    selected_model_class = MODEL_CLASSES.get(args.model_type, KneserNeyInterpolated)

    # --- Action: Check Data ---
    data_ready = False
    if args.action == "check_data":
        print("\n=== Action: Check Data Existence ===")
        if check_data_exists():
            print("--- Data Check Successful: Required data found. ---")
        else:
            print("--- Data Check Failed: Required data not found. Please ensure 'data/Train_Full' and 'data/vn_syllables.txt' are correctly populated. ---")
        print("--- Data Check Finished ---")
        return
    elif args.action == "train":
        print("\n=== Checking Data Existence for Training ===")
        if check_data_exists():
            print("--- Data Check Successful: Required data found. ---")
            data_ready = True
        else:
            print("--- Data Check Failed: Required data not found. ---")
            print("Please ensure 'data/Train_Full' and 'data/vn_syllables.txt' are correctly populated.")
            print("You can try running with --action check_data for more details or manually place the files.")
            return # Exit if data is not found for training

    # --- Action: Train Model ---
    if args.action == "train":
        if not data_ready:
            print("\nCannot proceed with training: Required data is not available. Please run --action check_data or ensure data is present.")
            return
        
        print("\n=== Action: Train Model ===")
        print("Loading corpus...")
        corpus = data_loader.load_corpus()
        if corpus:
            print(f"Training {args.model_type} model with N={args.n_gram}...")
            trained_model_instance = model_trainer.train_ngram_model(corpus, n=args.n_gram, model_class=selected_model_class)
            if trained_model_instance:
                model_trainer.save_model(trained_model_instance, filename=args.model_file)
            else:
                print("Model training failed.")
        else:
            print("Corpus not loaded. Skipping training.")
    
    print("\n--- Trainer Finished ---")

if __name__ == "__main__":
    main()