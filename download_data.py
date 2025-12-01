import os
import sys
import subprocess
import zipfile
import shutil
from config import * # Importa todas as variáveis de configuração

def download_and_setup_data():
    """
    Baixa o dataset do Kaggle, descompacta e move as pastas 'train' e 'test'
    para a estrutura final 'data/'.
    """
    # 1. VERIFICAÇÃO RÁPIDA
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        print("✅ Dados já estão na estrutura 'data/'. Pulando o setup.")
        return

    print("--- INICIANDO SETUP DE DADOS ---")
    
    # Criação de Pastas de trabalho (data é o destino final)
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
    print(f"Pastas de trabalho criadas: {DATASET_PATH} e {TEMP_DOWNLOAD_DIR}")

    # 2. DOWNLOAD VIA KAGGLE API (CORREÇÃO DE EXECUÇÃO: Executa 'kaggle' diretamente)
    print(f"Baixando dataset: {KAGGLE_DATASET_SLUG}...")
    try:
        # O subprocess.run tenta encontrar o binário 'kaggle' no PATH,
        # o que é feito pelo venv quando ele está ativo.
        command_parts = [
            "kaggle", "datasets", "download", 
            "-d", KAGGLE_DATASET_SLUG, 
            "-p", TEMP_DOWNLOAD_DIR
        ]
        # Não usamos shell=True para maior segurança
        subprocess.run(command_parts, check=True, capture_output=True, text=True)
        print("Download concluído com sucesso.")
    except subprocess.CalledProcessError as e:
        print("\n❌ ERRO ao baixar o dataset do Kaggle.")
        print("Verifique se a API do Kaggle está instalada (pip install kaggle) e configurada (kaggle.json).")
        print(f"Detalhes: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ ERRO: O comando 'kaggle' não foi encontrado. Certifique-se de que o VENV está ativo.")
        sys.exit(1)

    # 3. DESCOMPACTAR O ARQUIVO ZIP
    zip_name = KAGGLE_DATASET_SLUG.split('/')[-1] + '.zip'
    zip_path = os.path.join(TEMP_DOWNLOAD_DIR, zip_name)
    
    if not os.path.exists(zip_path):
        print(f"❌ ERRO: Arquivo ZIP esperado '{zip_path}' não encontrado.")
        sys.exit(1)
        
    print(f"Descompactando {zip_name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DOWNLOAD_DIR)
    except Exception as e:
        print(f"❌ ERRO ao descompactar o arquivo ZIP: {e}")
        sys.exit(1)

    # 4. ESTRUTURAÇÃO E MOVIMENTAÇÃO DOS DADOS (CORREÇÃO DE LÓGICA)
    extracted_folder_name = KAGGLE_DATASET_SLUG.split('/')[-1]
    
    # Caminhos de Origem (dentro da pasta descompactada)
    source_train = os.path.join(TEMP_DOWNLOAD_DIR, 'train') 
    source_test = os.path.join(TEMP_DOWNLOAD_DIR, 'test')
    
    # Caminhos de Destino (para a pasta 'data')
    dest_train = os.path.join(DATASET_PATH, 'train')
    dest_test = os.path.join(DATASET_PATH, 'test')
    
    try:
        # Move a pasta 'train' (com conteúdo) para 'data/train'
        shutil.move(source_train, dest_train) 
        # Move a pasta 'test' (com conteúdo) para 'data/test'
        shutil.move(source_test, dest_test) 
        print("✅ Pastas 'train' e 'test' movidas com sucesso para 'data/'.")
    except FileNotFoundError:
        print("❌ ERRO: Pastas 'train' ou 'test' não foram encontradas no caminho esperado após descompactação.")
        print(f"Verificado em: {TEMP_DOWNLOAD_DIR}")
        print("Pode haver uma subpasta intermediária diferente de 'libras'. Verifique o conteúdo do ZIP.")
        sys.exit(1)

    # 5. LIMPEZA
    shutil.rmtree(TEMP_DOWNLOAD_DIR)
    print("Limpeza de arquivos temporários concluída.")
    print("--- SETUP DE DADOS CONCLUÍDO ---\n")

if __name__ == "__main__":
    download_and_setup_data()