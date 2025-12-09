# HandTracker - Tradutor de LIBRAS com Inteligência Artificial

Este repositório contém um sistema híbrido de reconhecimento de **Língua Brasileira de Sinais (LIBRAS)**, dividido em duas abordagens distintas de Inteligência Artificial:

1.  **Modelo Dinâmico (Gestos em Vídeo):** Capaz de entender movimentos e frases (ex: gestos que dependem de movimento corporal) utilizando **MediaPipe Holistic** e redes **LSTM**.
2.  **Modelo Estático (Alfabeto em Imagem):** Capaz de reconhecer letras do alfabeto (A-Y) utilizando **MobileNetV2** e Transfer Learning.

---

## 📋 Pré-requisitos

* Python 3.8 ou superior
* Webcam funcional
* Conta no Kaggle (para baixar o dataset do modelo estático)

---

## 🛠️ Instalação Geral

Recomenda-se criar um ambiente virtual para evitar conflitos de versões.

```bash
# Clone o repositório
git clone [https://github.com/seu-usuario/HandTracker.git](https://github.com/seu-usuario/HandTracker.git)
cd HandTracker

# Crie um ambiente virtual (Opcional, mas recomendado)
python -m venv venv

# Ative o ambiente
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
````

## 🚀 1. Modelo Dinâmico (Reconhecimento de Gestos)

Este módulo foca no reconhecimento de sinais que envolvem movimento (vídeo). Ele extrai pontos-chave do corpo (pose) e das mãos e analisa a sequência temporal.

### 📂 Estrutura da Pasta `Dinâmico/`

  * `keypointsExtraction.py`: Processa vídeos brutos e extrai coordenadas (features).
  * `aumentar_dados.py`: Cria variações dos dados (zoom, ruído) para melhorar o treino.
  * `treinar_final.py`: Treina a rede neural (LSTM).
  * `testar_libras.py`: Tradutor em tempo real via webcam.
  * `avaliar_modelo.py`: Gera métricas e matriz de confusão.

### 👣 Passo a Passo para Uso

1.  **Instale as dependências:**

    ```bash
    pip install -r Dinâmico/requirements.txt
    ```

2.  **Prepare os Dados (Se tiver vídeos novos):**
    Coloque seus vídeos organizados em pastas (ex: `separados/Ola`, `separados/Obrigado`) e execute:

    ```bash
    cd Dinâmico
    python keypointsExtraction.py
    ```

    *Isso criará arquivos `.npy` na pasta `dataset_features`.*

3.  **Aumente o Dataset (Data Augmentation):**
    Para tornar o modelo mais robusto:

    ```bash
    python aumentar_dados.py
    ```

4.  **Treine o Modelo:**

    ```bash
    python treinar_final.py
    ```

    *Isso gerará o arquivo `libras_model.keras` e `actions.npy`.*

5.  **Teste em Tempo Real:**
    Para ver a tradução a acontecer na sua webcam:

    ```bash
    python testar_libras.py
    ```

    *Pressione 'q' para sair.*

6.  **Avalie a Performance:**
    Para ver a acurácia e a Matriz de Confusão:

    ```bash
    python avaliar_modelo.py
    ```

-----

## 📷 2. Modelo Estático (Alfabeto Manual)

Este módulo foca na classificação de imagens estáticas (frames individuais) para reconhecer as letras do alfabeto de LIBRAS.

### 📂 Estrutura da Pasta `Estático/`

  * `app.py`: Aplicação principal com Menu (Webcam ou Upload de Imagem).
  * `download_data.py`: Baixa o dataset do Kaggle automaticamente.
  * `config.py`: Configurações globais (caminhos, parâmetros).

### 👣 Passo a Passo para Uso

1.  **Instale as dependências:**

    ```bash
    pip install -r Estático/requirements.txt
    ```

2.  **Configuração do Dataset:**
    Este projeto usa o dataset `williansoliveira/libras` do Kaggle.

      * Crie uma chave de API no Kaggle e coloque o arquivo `kaggle.json` na pasta do seu utilizador (`~/.kaggle/` ou `%USERPROFILE%/.kaggle/`).
      * Execute o script de download:

    <!-- end list -->

    ```bash
    cd Estático
    python download_data.py
    ```

3.  **Execute o Aplicativo:**
    O app abrirá um menu interativo no terminal.

    ```bash
    python app.py
    ```

    **Opções do Menu:**

      * **1. Tradução em Tempo Real:** Abre a webcam e desenha um box verde. Coloque a mão dentro do box para traduzir a letra.
      * **2. Traduzir uma Imagem:** Abre uma janela para selecionar um arquivo de imagem (`.jpg`, `.png`) do computador e exibe a predição.

-----

## 🧠 Tecnologias Utilizadas

  * **Linguagem:** Python
  * **Visão Computacional:** OpenCV, MediaPipe
  * **Deep Learning:** TensorFlow, Keras
  * **Arquiteturas:**
      * *Dinâmico:* LSTM Bidirecional (Long Short-Term Memory)
      * *Estático:* MobileNetV2 (Transfer Learning)
  * **Manipulação de Dados:** NumPy, Pandas, Scikit-Learn

-----

## 🤝 Desenvolvedores

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/AndreyRodr">
        <img src="https://avatars.githubusercontent.com/u/134998417?v=4" width="100px;" alt="Foto do Andrey"/><br>
        <sub>
          <b>Andrey Rodrigues</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/LucassTeixeiraN">
        <img src="https://avatars.githubusercontent.com/u/82536301?v=4" width="100px;" alt="Foto do Lucas"/><br>
        <sub>
          <b>Lucas Teixeira</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/jptrava">
        <img src="https://avatars.githubusercontent.com/u/164881489?v=4" width="100px;" alt="Foto do João"/><br>
        <sub>
          <b>João Pedro Andrade</b>
        </sub>
      </a>
    </td>
  </tr>
</table>
