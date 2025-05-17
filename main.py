# main.py
import argparse
import os
import nltk

from utils import data_loader # Keep for SYLLABLES_PATH
from utils import predictor
# from utils.data_loader import check_data_exists # Not strictly needed if main only predicts
from nltk.lm import KneserNeyInterpolated, Laplace, WittenBellInterpolated # Keep for model loading

MODEL_CLASSES = {
    "KneserNeyInterpolated": KneserNeyInterpolated,
    "Laplace": Laplace,
    "WittenBellInterpolated": WittenBellInterpolated
}

def ensure_nltk_punkt():
    """Ensures NLTK 'punkt' tokenizer models are available."""
    try:
        nltk.data.find('tokenizers/punkt')
        # print("NLTK 'punkt' resource found.") # Less verbose
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' resource not found. Downloading...")
        nltk.download('punkt', quiet=True)
        print("'punkt' downloaded successfully.")
    except ImportError:
        print("NLTK library not found. Please install it: pip install nltk")
        exit(1)
    except Exception as e:
        print(f"An error occurred with NLTK 'punkt': {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Vietnamese Accent Predictor CLI")
    parser.add_argument("--model_file", type=str, default="kneserney_trigram_model.pkl",
                        help="Model filename to load for prediction.")
    parser.add_argument("--beam_k", type=int, default=3, # Keep beam_k=3 to get top 3 internally
                        help="Beam width (k) for accent prediction.")
    # text_input is now handled by input()

    args = parser.parse_args()

    print("--- Vietnamese Accent Predictor CLI ---")
    ensure_nltk_punkt()

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_script_dir, 'models')
    if not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' not found. Please run the training script first.")
        exit(1)

    model_path = os.path.join(model_dir, args.model_file)

    print(f"\nLoading model from {model_path}...")
    trained_model_instance = predictor.load_model(model_path)
    
    if not trained_model_instance:
        print(f"Could not load model. Ensure '{model_path}' exists and is valid.")
        print("You might need to run the training script first (e.g., train_model.py).")
        exit(1)
    else:
        print("Model loaded successfully.")

    syllables_file_path = data_loader.SYLLABLES_PATH 
    if not os.path.exists(syllables_file_path):
         print(f"Warning: Syllables file {syllables_file_path} not found. Accent prediction may be impaired.")
         print("Please ensure 'data/vn_syllables.txt' is correctly populated (e.g. by running train_model.py --action check_data).")

    print("\n--- Ready to Predict Accents ---")
    try:
        while True:
            text_input = input("Nhập câu tiếng Việt không dấu (hoặc gõ 'exit' để thoát): ")
            if text_input.lower() == 'exit':
                break
            if not text_input.strip():
                print("Vui lòng nhập một câu.")
                continue

            print(f"Đang dự đoán cho: '{text_input}'")
            # beam_search_predict_accents will still return up to k predictions
            predictions = predictor.beam_search_predict_accents(
                text_input, 
                trained_model_instance, 
                k=args.beam_k, 
                syllables_file=syllables_file_path
            )

            if predictions:
                print("Các dự đoán hàng đầu:")
                for i, (sent, score) in enumerate(predictions):
                    print(f"{i+1}. '{sent}' (Score: {score:.4f})")
            else:
                print("Không có dự đoán nào được trả về.")
            print("---") # Separator for next input

    except KeyboardInterrupt:
        print("\nĐã thoát chương trình.")
    finally:
        print("\n--- Predictor CLI Finished ---")

if __name__ == "__main__":
    main()