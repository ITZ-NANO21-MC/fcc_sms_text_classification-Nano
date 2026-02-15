import subprocess

# import libraries
try:
  subprocess.run(["pip", "install", "tensorflow"])
except Exception:
  pass

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(f"Versión de TensorFlow: {tf.__version__}")

# get data files
subprocess.run(["wget", "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"])
subprocess.run(["wget", "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"])

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Función para cargar y preparar datos
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df['message'].values, df['label'].values

# Cargar datos
train_texts, train_labels = load_data(train_file_path)
test_texts, test_labels = load_data(test_file_path)

# Paso 4: Preprocesamiento de texto
# Tokenización: Convertir palabras en números
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

# Convertir textos a secuencias numéricas
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Rellenar secuencias para que tengan la misma longitud
max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Paso 5: Construir el modelo de clasificación
def build_model():
    model = Sequential([
        # Capa de embedding: convierte índices en vectores densos
        Embedding(
            input_dim=10000,  # Tamaño del vocabulario
            output_dim=128,    # Dimensionalidad del embedding
            input_length=max_length  # Longitud de las secuencias
        ),

        # Capa LSTM bidireccional para capturar contexto
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),  # Regularización para prevenir sobreajuste

        # Segunda capa LSTM
        Bidirectional(LSTM(32)),
        Dropout(0.5),

        # Capa densa final con activación sigmoide
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',  # Para clasificación binaria
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

# Crear modelo
model = build_model()
model.summary()

# Paso 6: Entrenar el modelo
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_padded,
    train_labels,
    epochs=20,
    validation_split=0.2,
    callbacks=[early_stop],
    batch_size=64,
    verbose=1
)

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])

# Paso 7: Crear función de predicción
def predict_message(pred_text):
    # Preprocesar el texto de entrada
    sequence = tokenizer.texts_to_sequences([pred_text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Predecir probabilidad
    prediction_prob = model.predict(padded, verbose=0)[0][0]

    # Determinar etiqueta
    prediction_label = 'spam' if prediction_prob > 0.5 else 'ham'

    return [prediction_prob, prediction_label]

# Probar la función
pred_text = "how are you doing today?"
prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
