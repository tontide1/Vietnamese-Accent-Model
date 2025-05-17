# Vietnamese Trigram Language Model

This project implements a Trigram language model for Vietnamese, capable of predicting accents and generating text.

## Project Structure

```
ML_ML/
├── data/                  # Stores downloaded and processed data
├── models/                # Stores trained model files
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Handles data downloading and preprocessing
│   ├── model_trainer.py   # Trains the N-gram model
│   ├── predictor.py       # Uses the model for predictions
│   └── utils.py           # Utility functions
├── main.py                # Main script to run the pipeline
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Setup

1.  **Clone the repository (if applicable)**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    ```

5.  **Download training data and VN syllables:**
    The `data_loader.py` script will handle downloading the necessary data when run.
    Alternatively, you can manually download:
    *   Training data: [https://github.com/hoanganhpham1006/Vietnamese_Language_Model/raw/master/Train_Full.zip](https://github.com/hoanganhpham1006/Vietnamese_Language_Model/raw/master/Train_Full.zip) (and unzip to `data/Train_Full`)
    *   VN Syllables: [https://gist.githubusercontent.com/hieuthi/0f5adb7d3f79e7fb67e0e499004bf558/raw/135a4d9716e49a981624474156d6f247b9b46f6a/all-vietnamese-syllables.txt](https://gist.githubusercontent.com/hieuthi/0f5adb7d3f79e7fb67e0e499004bf558/raw/135a4d9716e49a981624474156d6f247b9b46f6a/all-vietnamese-syllables.txt) (save as `data/vn_syllables.txt`)

## Usage

Run the main script:
```bash
python main.py
```
This will typically perform:
1. Data loading and preprocessing.
2. Model training (and saving the model).
3. Example usage (e.g., accent prediction or text generation).

Refer to `main.py` and the individual modules in `src/` for more details.