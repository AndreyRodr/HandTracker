import cv2
import numpy as np
import os
import mediapipe as mp

# --- CONFIGURAÇÕES ---
# A pasta onde você organizou os vídeos (resultado do script anterior)
DATA_PATH = os.path.join('separados') 

# Onde vamos salvar os dados numéricos (numpy arrays)
EXPORT_PATH = os.path.join('dataset_features') 

# Inicializando o MediaPipe
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
    # Extrai coordenadas ou preenche com zero se não detectar nada
    # Pose (Corpo)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Rosto (Face) - Vamos pegar apenas o contorno ou usar tudo (468 pontos é muito, mas aqui pegaremos tudo para garantir)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # Mão Esquerda
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Mão Direita
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

def process_dataset():
    # Pega as pastas (labels) criadas (ex: Abacaxi, Banana...)
    actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            action_path = os.path.join(DATA_PATH, action)
            videos = os.listdir(action_path)
            
            # Cria pasta de destino para esta ação
            target_dir = os.path.join(EXPORT_PATH, action)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            for video_name in videos:
                video_path = os.path.join(action_path, video_name)
                cap = cv2.VideoCapture(video_path)
                
                # Vamos salvar cada vídeo como uma sequência de frames
                frames_data = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Detectar e extrair
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    frames_data.append(keypoints)
                
                cap.release()
                
                # Salva o vídeo processado como um arquivo numpy (.npy)
                # O nome do arquivo será o mesmo do vídeo, mas com extensão .npy
                npy_path = os.path.join(target_dir, video_name.split('.')[0])
                np.save(npy_path, np.array(frames_data))
                
                print(f"Processado: {action}/{video_name} -> Frames: {len(frames_data)}")

if __name__ == "__main__":
    if not os.path.exists(EXPORT_PATH):
        os.makedirs(EXPORT_PATH)
    process_dataset()
    print("Extração de características concluída!")