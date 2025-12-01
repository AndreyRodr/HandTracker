import os

# --- Configurações de Caminho e Kaggle ---
KAGGLE_DATASET_SLUG = "williansoliveira/libras"
TEMP_DOWNLOAD_DIR = "temp_kaggle_download"
DATASET_PATH = "data"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "test")
MODEL_PATH = 'sequential_2.keras'
CLASSES_FILE = 'classes.txt'
# ----------------------------------------

# --- Configurações Gerais do Modelo ---
SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 128
EPOCHS = 20
# ----------------------------------------

# --- Configurações da Webcam ---
PRED_BOX_SIZE = 300