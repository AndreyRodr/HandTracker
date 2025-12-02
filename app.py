import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import sys
from collections import deque, Counter
import tkinter as tk
from tkinter import filedialog

# Importa configurações
from config import *
# --- CONFIGURAÇÕES DO APP ---
CONFIDENCE_THRESHOLD = 0.30
BUFFER_SIZE = 5
VOTING_THRESHOLD = 0.5
IMAGE_DISPLAY_WIDTH = 1000 # Largura alvo para exibir imagens estáticas

# --- 1. Carregar Modelo e Classes (Comum a ambos) ---
def load_assets():
    print("Carregando modelo... aguarde.")
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_FILE):
            print(f"\nERRO: Arquivos '{MODEL_PATH}' ou '{CLASSES_FILE}' não encontrados.")
            print("Execute 'python train_model.py' primeiro.")
            sys.exit(1)

        model = load_model(MODEL_PATH)
        with open(CLASSES_FILE, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]

        print(f"✅ Modelo carregado com {len(classes)} classes.")
        return model, classes

    except Exception as e:
        print(f"ERRO CRÍTICO: {e}")
        sys.exit(1)

# --- 2. Função de Predição Genérica ---
def predict_array(img_array, model, classes):
    # Pré-processamento (MobileNetV2 exige valores entre -1 e 1)
    processed_img = preprocess_input(img_array)

    predictions = model.predict(processed_img, verbose=0)[0]
    predicted_index = tf.argmax(predictions).numpy()

    if predicted_index < len(classes):
        return classes[predicted_index], predictions[predicted_index]
    return "DESCONHECIDO", 0.0

# --- 3. MODO 1: Webcam em Tempo Real ---
def run_webcam(model, classes):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Erro: Webcam não encontrada.")
        return

    # --- TENTA DEFINIR UMA RESOLUÇÃO MAIOR (HD 720p) ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # ---------------------------------------------------

    # Buffer e Variáveis
    prediction_buffer = deque(maxlen=BUFFER_SIZE)
    current_display_class = "NULL"
    display_confidence = 0.0
    color = (100, 100, 100)

    # Lê as dimensões reais que a câmera aceitou
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ROI (Caixa Verde) centralizada
    x1, y1 = (W - PRED_BOX_SIZE) // 2, (H - PRED_BOX_SIZE) // 2
    x2, y2 = x1 + PRED_BOX_SIZE, y1 + PRED_BOX_SIZE

    print(f"\n--- MODO WEBCAM INICIADO (Resolução: {W}x{H}) ---")
    print("Pressione 'q' para voltar ao menu.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # Garante que a ROI está dentro dos limites do frame
        y1_clamp = max(0, y1); y2_clamp = min(H, y2)
        x1_clamp = max(0, x1); x2_clamp = min(W, x2)
        roi = frame[y1_clamp:y2_clamp, x1_clamp:x2_clamp]

        if roi.size == 0: continue

        # Preparar imagem para predição
        img_resized = cv2.resize(roi, IMG_SIZE)
        img_array = np.expand_dims(img_resized, axis=0)

        # Predição
        pred_class, conf = predict_array(img_array, model, classes)

        # Lógica de Estabilização (Votação)
        if conf < CONFIDENCE_THRESHOLD:
            current_display_class = "NULL"
            display_confidence = 0.0
            color = (100, 100, 100)
            prediction_buffer.clear()
        else:
            prediction_buffer.append(pred_class)
            if len(prediction_buffer) > 0:
                most_common = Counter(prediction_buffer).most_common(1)[0]
                if most_common[1] / len(prediction_buffer) >= VOTING_THRESHOLD:
                    current_display_class = most_common[0]
                    display_confidence = conf
                    color = (0, 255, 0)
                else:
                    color = (0, 255, 255)

        # Desenhar na tela
        cv2.rectangle(frame, (x1_clamp, y1_clamp), (x2_clamp, y2_clamp), (0, 255, 0), 2)
        text = "NULL" if current_display_class == "NULL" else f"{current_display_class} ({display_confidence:.2f})"

        # Ajusta tamanho da fonte baseado na resolução
        font_scale = 1.2 if W >= 1280 else 0.8
        thickness = 3 if W >= 1280 else 2

        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(frame, "Pressione 'q' para sair", (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Tradutor Libras - Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 4. MODO 2: Tradução de Imagem (CORRIGIDO) ---
def run_image_upload(model, classes):
    print("\n--- MODO IMAGEM ---")
    print("Abrindo seletor de arquivos...")

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Selecione uma imagem de Libras",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not file_path:
        print("Nenhuma imagem selecionada.")
        return

    img = cv2.imread(file_path)
    if img is None:
        print("❌ Erro ao abrir a imagem.")
        return

    # Fazer predição na imagem original (redimensionada para o modelo)
    img_resized_for_model = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img_resized_for_model, axis=0)
    pred_class, conf = predict_array(img_array, model, classes)

    print(f"\nResultado: {pred_class} (Confiança: {conf:.2f})")

    # --- REDIMENSIONAR PARA EXIBIÇÃO COM MELHOR QUALIDADE ---
    h_orig, w_orig, _ = img.shape
    scale_factor = IMAGE_DISPLAY_WIDTH / w_orig
    new_h = int(h_orig * scale_factor)

    # Usa INTER_CUBIC para um upscaling mais suave
    display_img_final = cv2.resize(img, (IMAGE_DISPLAY_WIDTH, new_h), interpolation=cv2.INTER_CUBIC)
    # --------------------------------------------------------

    # --- DESENHAR TEXTO NA IMAGEM JÁ REDIMENSIONADA ---
    # Ajusta escala da fonte baseado na largura final da imagem
    font_scale = max(1, IMAGE_DISPLAY_WIDTH // 600)
    thickness = max(2, IMAGE_DISPLAY_WIDTH // 400)

    # Posição do texto (agora fixa e nítida)
    cv2.putText(display_img_final, f"Pred: {pred_class}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    cv2.putText(display_img_final, f"Conf: {conf:.2f}", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 0), thickness)
    # --------------------------------------------------

    cv2.imshow('Tradutor Libras - Imagem Estatica', display_img_final)
    print("Pressione qualquer tecla na janela da imagem para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 5. MENU PRINCIPAL ---
def main_menu():
    model, classes = load_assets()

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n===================================")
        print("   TRADUTOR DE LIBRAS (MobileNetV2)")
        print("===================================")
        print("1. Tradução em Tempo Real (Webcam)")
        print("2. Traduzir uma Imagem (Arquivo)")
        print("3. Sair")
        print("===================================")
        choice = input("Escolha uma opção (1-3): ").strip()

        if choice == '1':
            run_webcam(model, classes)
        elif choice == '2':
            run_image_upload(model, classes)
        elif choice == '3':
            print("Encerrando...")
            break
        else:
            print("Opção inválida!")
            import time
            time.sleep(1)

if __name__ == "__main__":
    main_menu()