import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model

# --- CONFIGURAÇÕES ---
DATA_PATH = os.path.join('dataset_features')
MODEL_PATH = 'libras_model.keras'
ACTIONS_PATH = 'actions.npy'
MAX_LENGTH = 30
# ---------------------

def processar_arquivo(caminho_arquivo):
    """
    Replica EXATAMENTE o pré-processamento feito no treino.
    """
    res = np.load(caminho_arquivo)
    
    # REMOVER ROSTO (Se tiver 1662 colunas, reduz para 258)
    if res.shape[1] > 258:
        pose = res[:, :132]
        hands = res[:, 1536:]
        res = np.concatenate([pose, hands], axis=1)
    
    # AMOSTRAGEM TEMPORAL (Transformar em 30 frames)
    total_frames = res.shape[0]
    target_frames = MAX_LENGTH
    
    if total_frames > target_frames:
        indices = np.linspace(0, total_frames-1, target_frames).astype(int)
        res = res[indices]
    elif total_frames < target_frames:
        if total_frames == 0:
            return None # Arquivo vazio
        padding = np.zeros((target_frames - total_frames, res.shape[1]))
        res = np.concatenate((res, padding))
        
    return res

def avaliar():
    # Carregar Modelo e Nomes
    if not os.path.exists(MODEL_PATH):
        print("Erro: Modelo não encontrado.")
        return

    print("Carregando modelo...")
    model = load_model(MODEL_PATH)
    actions = np.load(ACTIONS_PATH)
    label_map = {label:num for num, label in enumerate(actions)}
    
    print(f"Avaliando {len(actions)} classes: {actions}")

    y_true = [] # Resposta Correta (Gabarito)
    y_pred = [] # Resposta da IA (Chute)

    # Varrer todos os arquivos
    print("Iniciando varredura dos dados...")
    
    for action in actions:
        caminho_pasta = os.path.join(DATA_PATH, action)
        if not os.path.exists(caminho_pasta): continue
        
        arquivos = [f for f in os.listdir(caminho_pasta) if f.endswith('.npy')]
        
        for arquivo in arquivos:
            caminho_arq = os.path.join(caminho_pasta, arquivo)
            
            # Processa
            dados = processar_arquivo(caminho_arq)
            if dados is None: continue
            
            # Prepara para o modelo (1, 30, 258)
            dados_input = np.expand_dims(dados, axis=0)
            
            # Predição
            res = model.predict(dados_input, verbose=0)[0]
            
            # Guarda os resultados
            y_true.append(label_map[action]) # Índice correto
            y_pred.append(np.argmax(res))    # Índice que a IA chutou

    # Gerar Métricas
    print("\n" + "="*50)
    print("RELATÓRIO DE AVALIAÇÃO")
    print("="*50)
    
    acuracia = accuracy_score(y_true, y_pred)
    print(f"Acurácia Global: {acuracia*100:.2f}%")
    print("-" * 50)
    
    # Relatório Detalhado (Precision, Recall, F1)
    print("\nDesempenho por Gesto:")
    print(classification_report(y_true, y_pred, target_names=actions))
    
    # Matriz de Confusão Gráfica
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.title('Matriz de Confusão - LIBRAS')
    plt.ylabel('Verdadeiro (O que você fez)')
    plt.xlabel('Predito (O que a IA entendeu)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        import matplotlib
        import seaborn
        avaliar()
    except ImportError:
        print("Você precisa instalar matplotlib e seaborn para ver os gráficos.")
        print("Rode: pip install matplotlib seaborn scikit-learn")