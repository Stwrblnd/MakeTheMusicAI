import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
import pickle


def load_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().split('\n')
    chords = [line.split(' ') for line in lines if line]
    return chords

def normalize_chord_name(chord_name):
    normalization_map = {
        'A#': 'Bb', 'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab',
        'A#m': 'Bbm', 'C#m': 'Dbm', 'D#m': 'Ebm', 'F#m': 'Gbm', 'G#m': 'Abm'
    }
    # Нормализация аккорда
    return normalization_map.get(chord_name, chord_name)

def preprocess_data(chords):
    # Нормализуем все аккорды в прогрессии перед созданием словаря
    chords_normalized = [[normalize_chord_name(chord) for chord in progression] for progression in chords]
    unique_chords = set(chord for progression in chords_normalized for chord in progression)
    chord_to_int = {chord: i for i, chord in enumerate(unique_chords)}
    int_to_chord = {i: chord for chord, i in chord_to_int.items()}
    
    # После нормализации продолжаем с созданием последовательностей
    sequences = []
    for progression in chords_normalized:
        sequence = [chord_to_int[chord] for chord in progression]
        sequences.append(sequence)

    X = []
    y = []
    for sequence in sequences:
        for i in range(len(sequence) - 3):
            X.append(sequence[i:i+3])
            y.append(sequence[i+3])

    X = np.array(X)
    y = to_categorical(y, num_classes=len(unique_chords))
    return X, y, chord_to_int, int_to_chord

def create_model(num_unique_chords):
    model = Sequential([
        Embedding(input_dim=num_unique_chords, output_dim=100, input_length=3),  # Увеличенный output_dim
        LSTM(100, return_sequences=True),
        Dropout(0.2),  # Добавлен слой Dropout
        LSTM(100),
        Dense(num_unique_chords, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(file_path, model_name):
    chords = load_dataset(file_path)
    X, y, chord_to_int, int_to_chord = preprocess_data(chords)
    model = create_model(num_unique_chords=len(chord_to_int))
    model.fit(X, y, epochs=80, verbose=1)
    model.save(f'{model_name}.h5')
    with open(f'{model_name}_chord_to_int.pkl', 'wb') as f:
        pickle.dump(chord_to_int, f)
    with open(f'{model_name}_int_to_chord.pkl', 'wb') as f:
        pickle.dump(int_to_chord, f)


train_and_save_model('sad_chord_progressions.txt', 'sad_model')


