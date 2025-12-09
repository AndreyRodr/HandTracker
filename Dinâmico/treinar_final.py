import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- CONFIGURAÇÕES ---
DATA_PATH = os.path.join('dataset_features') 
MAX_LENGTH = 30 
BATCH_SIZE = 4
EPOCHS = 60 # Épocas suficientes para aprender 10 classes
# ---------------------

class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(30, 1662), n_classes=10):
        self.dim = (dim[0], 258)
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            # --- PROTEÇÃO CONTRA ARQUIVOS VAZIOS ---
            try:
                res = np.load(ID)
                # Se o arquivo estiver vazio ou com formato errado, força um erro
                if res.ndim < 2 or res.shape[0] == 0:
                    raise ValueError("Arquivo vazio")
            except Exception:
                # Se der erro, cria um vídeo 'fake' de zeros para não travar o treino
                # Isso é melhor do que crashar o programa
                res = np.zeros((30, 1662)) # Tamanho padrão do MediaPipe
            # ---------------------------------------

            # 1. REMOVER ROSTO (Lógica original)
            # Verifica se tem colunas suficientes antes de tentar cortar
            if res.shape[1] > 258:
                pose = res[:, :132]
                hands = res[:, 1536:]
                res = np.concatenate([pose, hands], axis=1)
            
            # 2. AMOSTRAGEM TEMPORAL
            total_frames = res.shape[0]
            target_frames = self.dim[0]
            
            if total_frames > target_frames:
                indices = np.linspace(0, total_frames-1, target_frames).astype(int)
                res = res[indices]
            elif total_frames < target_frames:
                # Se o vídeo for vazio (0 frames), cria array de zeros do tamanho certo
                if total_frames == 0:
                    res = np.zeros((target_frames, self.dim[1]))
                else:
                    padding = np.zeros((target_frames - total_frames, res.shape[1]))
                    res = np.concatenate((res, padding))
            
            X[i,] = res
            y[i] = self.labels[ID]

        return X, to_categorical(y, num_classes=self.n_classes)


def treinar():
    # 1. PEGA AS 10 PRIMEIRAS CLASSES
    all_folders = sorted([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
    actions = np.array(all_folders[:10]) 
    
    print(f"🚀 TREINANDO COM: {actions}")
    
    # SALVA OS NOMES AGORA (Para garantir sincronia)
    np.save('actions.npy', actions)
    print("✅ actions.npy salvo!")

    label_map = {label:num for num, label in enumerate(actions)}
    file_paths = []
    file_labels = {}
    
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        for file_name in os.listdir(action_path):
            if file_name.endswith('.npy'):
                full_path = os.path.join(action_path, file_name)
                file_paths.append(full_path)
                file_labels[full_path] = label_map[action]

    # Divisão
    X_train_paths, X_test_paths = train_test_split(file_paths, test_size=0.20)
    
    # Geradores
    input_shape = (MAX_LENGTH, 258) 
    training_generator = DataGenerator(X_train_paths, file_labels, batch_size=BATCH_SIZE, dim=input_shape, n_classes=len(actions))
    validation_generator = DataGenerator(X_test_paths, file_labels, batch_size=BATCH_SIZE, dim=input_shape, n_classes=len(actions))

    # Modelo
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Masking(mask_value=0.0))
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh')))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    # Checkpoint (Salva o melhor) e EarlyStopping (Para se parar de aprender)
    checkpoint = ModelCheckpoint('libras_model.keras', monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True)

    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print("Iniciando treino...")
    model.fit(
        training_generator, 
        epochs=EPOCHS, 
        validation_data=validation_generator, 
        callbacks=[checkpoint, early_stop]
    )
    print("🏁 Treino concluído.")

if __name__ == "__main__":
    treinar()