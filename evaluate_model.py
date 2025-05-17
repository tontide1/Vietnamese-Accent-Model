import os
import json
from tqdm import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer
import multiprocessing as mp
import pickle
import gc

# Add the directory containing the 'utils' package to sys.path
# This assumes your 'utils' directory is in the same directory as this notebook file,
# or in a directory that can be navigated to from the current working directory.
# Adjust the 'project_root' calculation if your 'utils' directory is located elsewhere.
import sys
import os

# Get the directory of the current notebook file
# This might vary slightly depending on the environment, but in Colab,
# the notebook's path can often be determined relative to the current working directory.
# Let's assume the 'utils' directory is in the same parent directory as the notebook.
notebook_dir = os.getcwd() # Get current working directory
project_root = notebook_dir # Assuming 'utils' is directly in the notebook's directory

# Add the project root (or the directory containing 'utils') to sys.path
# Adjust 'project_root' if your 'utils' directory is located elsewhere relative to the notebook
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to sys.path")


from utils.data_loader import load_and_split_corpus, TRAIN_EXTRACT_PATH, SYLLABLES_PATH
from utils.predictor import load_model, beam_search_predict_accents
from utils.model_trainer import DEFAULT_MODEL_FILENAME

# --- Configuration ---
# Update MODEL_PATH and SYLLABLES_PATH if they are not relative to the project root
MODEL_PATH_ORIGINAL_SETUP = os.path.join("models", DEFAULT_MODEL_FILENAME)
# Assuming SYLLABLES_PATH is also relative to the project root
# If SYLLABLES_PATH comes from data_loader, ensure data_loader uses paths relative to project_root
# or adjust it here if needed.
# SYLLABLES_PATH = os.path.join(project_root, "data", "syllables.txt") # Example if it's a fixed path

RESULTS_FILE = "evaluation_results.json"
TEST_SET_SIZE = 0.2
RANDOM_SEED = 42
BEAM_K = 3

# --- RAM Management for Model Loading by Workers (KHÔI PHỤC) ---
TARGET_RAM_FOR_MODELS_GB = 24.0 # Mục tiêu 24GB cho các model của worker (theo yêu cầu user)
ASSUMED_MODEL_SIZE_GB = 0.2  # Đặt lại thành 0.2 GB (200MB) cho model 150MB
# ABSOLUTE_MAX_WORKERS = os.cpu_count() or 8     # Giới hạn cứng số worker, ví dụ bằng số core CPU hoặc một giá trị an toàn (8)
ABSOLUTE_MAX_WORKERS = 4 

_detokenizer = TreebankWordDetokenizer()

# Biến toàn cục để lưu model cho mỗi worker, được khởi tạo là None
# Mỗi worker sẽ cố gắng tải model một lần
_worker_model_cache = {}

def _get_model_for_worker(model_path_arg: str):
    pid = os.getpid()
    if pid not in _worker_model_cache or _worker_model_cache[pid] is None or _worker_model_cache[pid] == "ERROR":
        if pid not in _worker_model_cache or _worker_model_cache[pid] != "ERROR":
             print(f"Prediction Worker {pid} attempting to load model from {model_path_arg}...")
        
        model = load_model(model_path_arg)
        if model is None:
            # Lỗi CRITICAL đã được in bởi load_model nếu có
            _worker_model_cache[pid] = "ERROR"
        else:
            _worker_model_cache[pid] = model
            print(f"Prediction Worker {pid} successfully loaded model.")
    
    current_model_status = _worker_model_cache[pid]
    if current_model_status == "ERROR":
        return None
    return current_model_status

def _predict_item_parallel(item_data: tuple):
    unaccented_input_str, tokenized_true_accented_sent = item_data
    worker_detokenizer = TreebankWordDetokenizer()
    true_accented_str = worker_detokenizer.detokenize(tokenized_true_accented_sent)

    # Lấy (hoặc tải nếu chưa có) model cho worker này
    # MODEL_PATH_ORIGINAL_SETUP là biến toàn cục từ scope ngoài
    current_model = _get_model_for_worker(MODEL_PATH_ORIGINAL_SETUP)

    if current_model is None:
        # Lỗi đã được print trong _get_model_for_worker
        return {
            "input_unaccented": unaccented_input_str,
            "true_accented": true_accented_str,
            "predicted_accented": "WORKER_MODEL_LOAD_ERROR"
        }

    predictions = beam_search_predict_accents(
        text_no_accents=unaccented_input_str,
        model=current_model,
        k=BEAM_K,
        syllables_file=SYLLABLES_PATH, # SYLLABLES_PATH is from utils.data_loader
        detokenizer=worker_detokenizer
    )

    predicted_accented_str = predictions[0][0] if predictions else ""

    return {
        "input_unaccented": unaccented_input_str,
        "true_accented": true_accented_str,
        "predicted_accented": predicted_accented_str
    }

# --- Metrics Calculation ---
def calculate_sentence_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct_sentences = 0
    for res in results:
        if res["true_accented"].strip() == res["predicted_accented"].strip():
            correct_sentences += 1
    return (correct_sentences / len(results)) * 100

def calculate_word_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    total_words = 0
    correct_words = 0
    for res in results:
        true_words = res["true_accented"].strip().split()
        predicted_words = res["predicted_accented"].strip().split()
        len_min = min(len(true_words), len(predicted_words))
        for i in range(len_min):
            if true_words[i] == predicted_words[i]:
                correct_words += 1
        total_words += len(true_words)
    if total_words == 0: return 0.0
    return (correct_words / total_words) * 100

from nltk.metrics.distance import edit_distance

def calculate_cer(results: list[dict]) -> float:
    if not results:
        return 0.0
    total_edit_distance = 0
    total_true_chars = 0
    for res in results:
        true_str = res["true_accented"].strip()
        pred_str = res["predicted_accented"].strip()
        if not true_str and not pred_str:
            dist = 0
        elif not true_str:
            dist = len(pred_str)
        elif not pred_str:
            dist = len(true_str)
        else:
            dist = edit_distance(pred_str, true_str)
        total_edit_distance += dist
        total_true_chars += len(true_str)
    if total_true_chars == 0: return 1.0 if total_edit_distance > 0 else 0.0
    return (total_edit_distance / total_true_chars) * 100

# --- Visualization (Placeholder - Requires matplotlib) ---
def display_sample_results(results: list[dict], num_samples: int = 5):
    print("\n--- Sample Predictions ---")
    for i, res in enumerate(results[:num_samples]):
        print(f"Sample {i+1}:")
        print(f"  Input:     '{res['input_unaccented']}'")
        print(f"  True:      '{res['true_accented']}'")
        print(f"  Predicted: '{res['predicted_accented']}'")
        print("---")

def plot_metrics(metrics: dict):
    """Plots metrics using matplotlib. Requires matplotlib to be installed."""
    try:
        import matplotlib.pyplot as plt
        names = list(metrics.keys())
        values = list(metrics.values())

        plt.figure(figsize=(10, 5))
        plt.bar(names, values)
        plt.ylabel('Percentage (%)')
        plt.title('Model Evaluation Metrics')

        # Thêm giá trị trên mỗi cột
        for i, value in enumerate(values):
            plt.text(i, value + 0.5, f"{value:.2f}%", ha = 'center')

        # Đảm bảo thư mục plots tồn tại
        plots_dir = "plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plot_path = os.path.join(plots_dir, "evaluation_metrics.png")
        plt.savefig(plot_path)
        print(f"\nMetrics plot saved to {plot_path}")
        # plt.show() # Bỏ comment nếu muốn hiển thị trực tiếp
    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation. Please install it: pip install matplotlib")
    except Exception as e:
        print(f"\nError plotting metrics: {e}")


def main():
    print("--- Model Evaluation Script ---")

    evaluation_results = []
    # Prepend project_root to RESULTS_FILE if it should be saved there
    results_file_path = os.path.join(project_root, RESULTS_FILE)
    if os.path.exists(results_file_path):
        print(f"\nFound existing results file: {results_file_path}. Loading them.")
        try:
            with open(results_file_path, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
            print(f"Loaded {len(evaluation_results)} results from file.")
        except Exception as e:
            print(f"Error loading results file: {e}. Will re-run predictions.")
            evaluation_results = []

    if not evaluation_results:
        print("\nStep 1: Loading test data (as results file not found or failed to load)...")
        # Ensure load_and_split_corpus uses TRAIN_EXTRACT_PATH which should now be accessible
        # If TRAIN_EXTRACT_PATH is a relative path defined within utils.data_loader,
        # ensure it's correctly resolved relative to the project root inside that module.
        # If it's defined globally here, you might need to adjust it.
        _, test_set = load_and_split_corpus(
            data_extract_path=TRAIN_EXTRACT_PATH, # Assuming this path is correct or relative to project_root
            test_size=TEST_SET_SIZE,
            random_seed=RANDOM_SEED
        )
        if not test_set:
            print("Test set is empty. Cannot proceed with evaluation.")
            return
        print(f"Loaded {len(test_set)} sentences for testing.")

        print(f"\nStep 2: Preparing for predictions on the test set using multiprocessing...")

        # Ensure SYLLABLES_PATH is also accessible or correctly resolved
        # If SYLLABLES_PATH is a relative path defined within utils.data_loader,
        # ensure it's correctly resolved relative to the project root inside that module.
        # If it's defined globally here, you might need to adjust it.
        # Example if SYLLABLES_PATH was defined relative to project_root:
        # syllables_path_abs = os.path.join(project_root, SYLLABLES_PATH)
        # if not os.path.exists(syllables_path_abs):
        #     print(f"CRITICAL ERROR: Syllables file not found at {syllables_path_abs}. Predictions will likely fail.")
        #     print("Please ensure the syllables file is available. Exiting.")
        #     return

        if not os.path.exists(SYLLABLES_PATH): # Assuming SYLLABLES_PATH is correctly defined/resolved by utils.data_loader
             print(f"CRITICAL ERROR: Syllables file not found at {SYLLABLES_PATH}. Predictions will likely fail.")
             print("Please ensure the syllables file is available. Exiting.")
             return

        # --- Pre-flight Model Load Check (KHÔI PHỤC) ---
        print(f"\n--- Pre-flight Model Load Check (Main Process) ---")
        main_process_model = load_model(MODEL_PATH_ORIGINAL_SETUP) # Sử dụng đường dẫn gốc
        if main_process_model is None:
            print(f"CRITICAL: Failed to load model in the main process from '{MODEL_PATH_ORIGINAL_SETUP}'. ")
            print("This indicates a problem with the model file itself or the loading function.")
            print("Please check the model file path (models/{DEFAULT_MODEL_FILENAME}) and integrity. Exiting.")
            return
        else:
            try:
                model_pickled_size_bytes = len(pickle.dumps(main_process_model))
                model_pickled_size_mb = model_pickled_size_bytes / (1024 * 1024)
                print(f"Model '{MODEL_PATH_ORIGINAL_SETUP}' loaded successfully. Approx. pickled size: {model_pickled_size_mb:.2f} MB.")
                print(f"Note: Actual RAM footprint when active can be larger.")
            except Exception as e_pickle_size:
                print(f"Model loaded, but could not estimate pickled size: {e_pickle_size}")
            del main_process_model 
            gc.collect() 
            print("Pre-flight check passed. Model is loadable. Memory cleared in main process.")

        # --- Tính toán số lượng worker (KHÔI PHỤC VÀ ĐIỀU CHỈNH) ---
        cpu_cores = os.cpu_count() or 1 # Đảm bảo ít nhất 1 core
        ram_based_workers = 1 
        if ASSUMED_MODEL_SIZE_GB > 0 and TARGET_RAM_FOR_MODELS_GB > 0:
            ram_based_workers = max(1, int(TARGET_RAM_FOR_MODELS_GB / ASSUMED_MODEL_SIZE_GB))
        
        # Số lượng worker sẽ là min của (số tính theo RAM, số core CPU, và giới hạn tuyệt đối)
        # Với model nhỏ, ram_based_workers có thể rất lớn, nên cpu_cores và ABSOLUTE_MAX_WORKERS sẽ là yếu tố quyết định chính
        num_processes = min(ram_based_workers, cpu_cores, ABSOLUTE_MAX_WORKERS) 
        
        # Đảm bảo num_processes luôn ít nhất là 1 nếu có test_set và không có lỗi nào khác
        if len(test_set) > 0 and num_processes == 0: 
            num_processes = 1
            
        print(f"Target RAM for worker models: {TARGET_RAM_FOR_MODELS_GB}GB. Assumed 1 model size: {ASSUMED_MODEL_SIZE_GB}GB.")
        print(f"Calculated max workers by RAM: {ram_based_workers}. CPU cores: {cpu_cores}. Absolute cap: {ABSOLUTE_MAX_WORKERS}.")
        print(f"Using {num_processes} worker process(es) for prediction.")

        if num_processes > 0:
            with mp.Pool(processes=num_processes) as pool:
                # Điều chỉnh chunksize, có thể tăng divisor nếu mỗi task nhẹ
                chunk_size = max(1, len(test_set) // (num_processes * 8)) 
                print(f"Calculated chunk size for prediction pool: {chunk_size}")

                results_iterator = pool.imap(
                    _predict_item_parallel, 
                    tqdm(test_set, desc=f"Evaluating with {num_processes} worker(s)"),
                    chunksize=chunk_size
                )
                evaluation_results = [res for res in results_iterator if res is not None]
        elif test_set: # Nếu num_processes = 0 (ví dụ do cấu hình RAM quá thấp) nhưng vẫn có test_set
            print("WARNING: Number of processes calculated to 0. Running sequentially in main process.")
            # Fallback to sequential processing if num_processes is 0 for some reason
            temp_model_for_sequential = load_model(MODEL_PATH_ORIGINAL_SETUP)
            if temp_model_for_sequential is None:
                print("CRITICAL: Failed to load model for sequential fallback. Exiting.")
                return
            for item in tqdm(test_set, desc="Evaluating (sequential fallback)"):
                evaluation_results.append(_predict_item_parallel(item)) # _predict_item_parallel sẽ dùng _get_model_for_worker
            del temp_model_for_sequential
            gc.collect()
        else: # No test set
             evaluation_results = [] # Ensure it's an empty list

        print(f"\nStep 3: Saving evaluation results to {results_file_path}...")
        try:
            with open(results_file_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved {len(evaluation_results)} results to {results_file_path}")
        except IOError as e:
            print(f"Error saving results to {results_file_path}: {e}")

    if not evaluation_results:
        print("No evaluation results to process. Exiting.")
        return

    model_load_errors = sum(1 for res in evaluation_results if res.get("predicted_accented") == "WORKER_MODEL_LOAD_ERROR")
    if model_load_errors > 0:
        print(f"\nWARNING: {model_load_errors}/{len(evaluation_results)} items had model loading errors in worker processes.")
        print("Metrics might be inaccurate. Please check worker logs/model path and consider:")
        print("  - Reducing ABSOLUTE_MAX_WORKERS.")
        print("  - Increasing ASSUMED_MODEL_SIZE_GB if workers fail to load the model (indicates model is larger than assumed).")
        print("  - Ensuring MODEL_PATH is correct and the model file is not corrupted.")

    print("\nStep 4: Calculating metrics...")
    sent_accuracy = calculate_sentence_accuracy(evaluation_results)
    word_acc = calculate_word_accuracy(evaluation_results)
    char_error_rate = calculate_cer(evaluation_results)
    char_accuracy = 100.0 - char_error_rate

    print(f"\n--- Evaluation Metrics ---")
    print(f"Sentence Accuracy: {sent_accuracy:.2f}%")
    print(f"Word Accuracy: {word_acc:.2f}%")
    print(f"Character Error Rate (CER): {char_error_rate:.2f}%")
    print(f"Character Accuracy: {char_accuracy:.2f}%")

    metrics_to_plot = {
        "Sentence Accuracy": sent_accuracy,
        "Word Accuracy": word_acc,
        "Character Accuracy": char_accuracy
    }

    print("\nStep 5: Displaying sample results...")
    display_sample_results(evaluation_results, num_samples=5)

    print("\nStep 6: Plotting metrics...")
    plot_metrics(metrics_to_plot)

    print("\n--- Evaluation Complete ---")

if __name__ == '__main__':
    # Quan trọng: Đảm bảo context cho multiprocessing được set đúng, đặc biệt trên Windows
    # Có thể cần 'spawn' hoặc 'forkserver' nếu 'fork' (mặc định trên Linux) gây vấn đề
    # mp.set_start_method('spawn', force=True) # Bỏ comment và thử nếu gặp lỗi liên quan đến start method
    main()