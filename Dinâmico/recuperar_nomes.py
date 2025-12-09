import numpy as np
import os

# --- CONFIGURAÇÃO ---
DATA_PATH = os.path.join('dataset_features')
QTD_CLASSES = 50 # O mesmo número que você usou no treino (MVP = 50)
# --------------------

def gerar_arquivo_nomes():
    if not os.path.exists(DATA_PATH):
        print(f"Erro: Pasta '{DATA_PATH}' não encontrada.")
        return

    # 1. Lê as pastas e ordena alfabeticamente (igual ao treino)
    todas_pastas = sorted([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
    
    # 2. Pega as primeiras 50 (ou o número que você treinou)
    actions = np.array(todas_pastas[:QTD_CLASSES])
    
    print(f"Recuperando nomes para {len(actions)} classes...")
    print(f"Primeira classe: {actions[0]}")
    print(f"Última classe: {actions[-1]}")
    
    # 3. Salva o arquivo que está faltando
    np.save('actions.npy', actions)
    print("✅ Sucesso! Arquivo 'actions.npy' criado.")

if __name__ == "__main__":
    gerar_arquivo_nomes()