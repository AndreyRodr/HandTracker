import numpy as np
import os

# Caminho dos dados extraídos
DATA_PATH = os.path.join('dataset_features') 

def add_noise(data, scale=0.02):
    """Adiciona um ruído aleatório (simula tremedeira ou imprecisão)"""
    noise = np.random.normal(loc=0.0, scale=scale, size=data.shape)
    return data + noise

def scale_data(data, scaling_factor=0.1):
    """Aumenta ou diminui os valores (simula zoom/distância)"""
    # Gera um fator aleatório entre 0.9 e 1.1 (por exemplo)
    factor = np.random.uniform(1 - scaling_factor, 1 + scaling_factor)
    return data * factor

def aumentar_dataset():
    actions = [name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))]
    
    total_gerado = 0
    
    print(f"Iniciando aumento de dados em {len(actions)} classes...")

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        files = [f for f in os.listdir(action_path) if f.endswith('.npy') and 'aug' not in f]
        
        # Se a pasta já tiver muitos arquivos (ex: já rodou antes), pule
        if len(files) == 0: continue

        for file_name in files:
            original_data = np.load(os.path.join(action_path, file_name))
            
            # --- Variação 1: Ruído Leve ---
            aug_noise = add_noise(original_data, scale=0.01)
            save_path = os.path.join(action_path, file_name.replace('.npy', '_aug_noise.npy'))
            np.save(save_path, aug_noise)
            
            # --- Variação 2: Ruído Forte ---
            aug_noise2 = add_noise(original_data, scale=0.03)
            save_path2 = os.path.join(action_path, file_name.replace('.npy', '_aug_noise2.npy'))
            np.save(save_path2, aug_noise2)

            # --- Variação 3: Escala (Zoom Out) ---
            aug_scale1 = scale_data(original_data, scaling_factor=0.05)
            save_path3 = os.path.join(action_path, file_name.replace('.npy', '_aug_scale1.npy'))
            np.save(save_path3, aug_scale1)
            
            # --- Variação 4: Escala (Zoom In/Out mais forte) ---
            aug_scale2 = scale_data(original_data, scaling_factor=0.10)
            save_path4 = os.path.join(action_path, file_name.replace('.npy', '_aug_scale2.npy'))
            np.save(save_path4, aug_scale2)
            
            total_gerado += 4 # Criamos 4 novos para cada 1 original

    print(f"Concluído! {total_gerado} novos arquivos de treino foram criados.")

if __name__ == "__main__":
    aumentar_dataset()