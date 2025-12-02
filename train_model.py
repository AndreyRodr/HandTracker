import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.client import device_lib
import subprocess
import zipfile
import shutil

# Importa configurações e define Workers
from config import * 
NUM_WORKERS = 4 
# ----------------------------------------

def setup_data():
    """Baixa e organiza o dataset se necessário."""
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        print("✅ Dados já estão na estrutura 'data/'. Pulando o setup.")
        return

    print("--- INICIANDO SETUP DE DADOS ---")
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

    print(f"Baixando dataset: {KAGGLE_DATASET_SLUG}...")
    try:
        command_parts = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET_SLUG, "-p", TEMP_DOWNLOAD_DIR]
        subprocess.run(command_parts, check=True, capture_output=True, text=True)
        print("Download concluído com sucesso.")
    except Exception as e:
        print(f"\n❌ ERRO no download: {e}")
        sys.exit(1)

    zip_name = KAGGLE_DATASET_SLUG.split('/')[-1] + '.zip'
    zip_path = os.path.join(TEMP_DOWNLOAD_DIR, zip_name)
    
    print(f"Descompactando {zip_name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DOWNLOAD_DIR)
    except Exception as e:
        print(f"❌ ERRO ao descompactar: {e}")
        sys.exit(1)

    # Move train/test para a pasta data
    try:
        shutil.move(os.path.join(TEMP_DOWNLOAD_DIR, 'train'), os.path.join(DATASET_PATH, 'train'))
        shutil.move(os.path.join(TEMP_DOWNLOAD_DIR, 'test'), os.path.join(DATASET_PATH, 'test'))
        print("✅ Pastas movidas com sucesso.")
    except Exception:
        print("❌ ERRO: Estrutura de pastas inesperada no ZIP.")
        sys.exit(1)

    shutil.rmtree(TEMP_DOWNLOAD_DIR)
    print("--- SETUP CONCLUÍDO ---\n")


def create_model(num_classes):
    """Cria modelo MobileNetV2 congelado para Fase 1."""
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False 

    model = Sequential([
        base_model, GlobalAveragePooling2D(),
        Dense(512, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', metrics=['accuracy']
    )
    return model

def run_training():
    setup_data()
    
    # Verificações básicas
    if not (os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR)):
        print("❌ ERRO: Dados não encontrados.")
        sys.exit(1)
        
    print(f"--- Treinamento: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE} ---")

    # Geradores de Dados
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',
        validation_split=0.2
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
    validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')
    test_generator = val_test_datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

    num_train_samples = train_generator.samples
    num_validation_samples = validation_generator.samples
    classes = list(train_generator.class_indices.keys())
    
    with open(CLASSES_FILE, 'w') as f:
        for cls in classes: f.write(f"{cls}\n")

    # --- FASE 1: TREINAMENTO DO TOPO ---
    model = create_model(len(classes))
    model.summary()
    
    callbacks_fase1 = [
        ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
    ]

    print("\n--- FASE 1: TREINANDO TOPO ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=num_train_samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=num_validation_samples // BATCH_SIZE,
        callbacks=callbacks_fase1,
        verbose=1
    )

    # --- FASE 2: FINE-TUNING ---
    print("\n--- FASE 2: FINE-TUNING (DESCONGELANDO) ---")
    model = load_model(MODEL_PATH)
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Congelar as primeiras 50 camadas (preserva features básicas)
    for layer in base_model.layers[:50]:
        layer.trainable = False

    # Compila com LR inicial de 1e-5, mas usaremos ReduceLROnPlateau para baixar se precisar
    model.compile(
        optimizer=Adam(learning_rate=0.00001), 
        loss='categorical_crossentropy', metrics=['accuracy']
    )
    
    TOTAL_EPOCHS = history.epoch[-1] + 15 # +15 épocas para ajuste fino
    
    callbacks_fase2 = [
        ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=6, verbose=1, restore_best_weights=True),
        # NOVO: Reduz LR se estagnar, ajudando a chegar no ótimo global
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    ]

    history_fine_tune = model.fit(
        train_generator,
        steps_per_epoch=num_train_samples // BATCH_SIZE,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history.epoch[-1],
        validation_data=validation_generator,
        validation_steps=num_validation_samples // BATCH_SIZE,
        callbacks=callbacks_fase2,
        verbose=1
    )

    # --- AVALIAÇÃO ---
    print("\n--- AVALIAÇÃO FINAL ---")
    final_model = load_model(MODEL_PATH)
    loss, acc = final_model.evaluate(test_generator)
    print(f"Acurácia Final: {acc:.4f}")

    # Matriz de Confusão
    Y_pred = final_model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_generator.classes, y_pred)
    
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão Final')
    plt.show()

if __name__ == "__main__":
    run_training()