import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import os
import sys
from config import * # Importa todas as configurações


CONFIDENCE_THRESHOLD = 0.75
# --- 1. Carregar Modelo e Classes ---
def load_assets():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_FILE):
            print(f"ERRO: Arquivos de modelo ({MODEL_PATH}) ou classes ({CLASSES_FILE}) não encontrados.")
            print("Por favor, execute 'python train_model.py' primeiro.")
            sys.exit(1)
            
        model = load_model(MODEL_PATH)
        print(f"Modelo carregado com sucesso de: {MODEL_PATH}")

        with open(CLASSES_FILE, 'r') as f:
            classes = [line.strip() for line in f if line.strip()] 
        print(f"Classes carregadas: {classes}")
        return model, classes

    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo ou classes. Detalhes: {e}")
        sys.exit(1)

# --- 2. Função de Predição ---
def predict_frame(frame, model, classes):
    img = cv2.resize(frame, IMG_SIZE)
    img_array = np.expand_dims(img, axis=0)
    processed_img = preprocess_input(img_array)

    predictions = model.predict(processed_img, verbose=0)[0]
    predicted_index = tf.argmax(predictions).numpy()
    
    predicted_class = classes[predicted_index] if predicted_index < len(classes) else "CLASSE DESCONHECIDA"
    confidence = predictions[predicted_index]

    return predicted_class, confidence

# --- 3. Execução em Tempo Real com Webcam ---
def real_time_translation(model, classes):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam. Verifique a câmera e as permissões.")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    box_w = PRED_BOX_SIZE
    box_h = PRED_BOX_SIZE

    x1 = (W - box_w) // 2
    y1 = (H - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    last_stable_class = "INICIANDO"
    print("\nIniciando tradução em tempo real. Coloque sua mão na caixa verde. Pressione 'q' para sair.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # Espelha o frame

        # Desenhar e recortar a Região de Interesse (ROI) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]

        predicted_class, confidence = predict_frame(roi, model, classes)

        # --- LÓGICA DO LIMIAR DE CONFIANÇA (ADICIONADA) ---
        if confidence >= CONFIDENCE_THRESHOLD:
            # Se a confiança for alta, atualiza a classe estável
            last_stable_class = predicted_class
            color = (0, 0, 255) # Vermelho (Alto Confiança)
        else:
            # Se a confiança for baixa, mantém a classe anterior
            color = (0, 255, 255) # Amarelo (Baixa Confiança)


        # Exibir o resultado
        text = f"{predicted_class} ({confidence:.2f})"
        cv2.putText(frame, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow('Libras Real-Time Translator', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model, classes = load_assets()
    real_time_translation(model, classes)