# --- MONKEY PATCH (Correção Protobuf) ---
import google.protobuf.symbol_database
def GetPrototype(self, descriptor):
    return self.GetSymbol(descriptor.full_name)
if not hasattr(google.protobuf.symbol_database.SymbolDatabase, 'GetPrototype'):
    google.protobuf.symbol_database.SymbolDatabase.GetPrototype = GetPrototype
# ----------------------------------------

import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- CONFIGURAÇÕES ---
THRESHOLD = 0.70  # Certeza mínima para mostrar o nome (70%)
# ---------------------

# 1. Carrega o modelo treinado e os nomes
print("Carregando modelo...")
model = load_model('libras_model.keras')
actions = np.load('actions.npy')
print(f"Modelo carregado! Classes: {actions}")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def extract_keypoints(results):
    # Extrai Pose (33 pontos * 4 coords = 132)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Extrai Mãos (21 pontos * 3 coords = 63 cada)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatena: Pose + Mão E + Mão D = 258 pontos
    # (Exatamente o formato que o modelo aprendeu no treino)
    return np.concatenate([pose, lh, rh])

# Variáveis
sequence = []
sentence = []
prev_action = None
texto_atual = "..."
last_5_predictions = []

cap = cv2.VideoCapture(1)

# Configura a janela para ser redimensionável
cv2.namedWindow('Tradutor LIBRAS', cv2.WINDOW_NORMAL)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Detecção MediaPipe
        image, results = mediapipe_detection(frame, holistic)
        
        # Desenhar esqueleto (FEEDBACK VISUAL)
        # Se você não vir as linhas coloridas na mão, a IA não funciona.
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Extração de dados
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Mantém apenas os últimos 30 frames

        # Lógica de Predição
        if len(sequence) == 30:
            # Prepara os dados (1, 30, 258)
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

            last_5_predictions.append(res)
            if len(last_5_predictions) > 5:
                last_5_predictions.pop(0)
            
            prediction_avg = np.mean(last_5_predictions, axis=0)
            
            best_idx = np.argmax(prediction_avg)
            confidence = prediction_avg[best_idx]
            current_action = actions[best_idx]

            # Só confirma se passar do limiar
            if confidence > THRESHOLD:
                texto_atual = current_action


        # Interface Bonita
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        
        # MUDANÇA: Agora mostramos apenas 'texto_atual', sem join ou listas
        cv2.putText(image, texto_atual, (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Tradutor LIBRAS', image)

        # 'q' para sair
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()