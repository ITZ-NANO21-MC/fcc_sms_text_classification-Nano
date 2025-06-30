{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ITZ-NANO21-MC/fcc_sms_text_classification-Nano/blob/V2/fcc_sms_text_classification.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8RZOuS9LWQvv"
      },
      "outputs": [],
      "source": [
        "# import libraries\n",
        "try:\n",
        "  # %tensorflow_version only exists in Colab.\n",
        "  !pip install tf-nightly\n",
        "except Exception:\n",
        "  pass\n",
        "\n",
        "import tensorflow as tf\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional\n",
        "from tensorflow.keras.optimizers import Adam\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "from tensorflow.keras.preprocessing.text import Tokenizer\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "\n",
        "print(f\"Versión de TensorFlow: {tf.__version__}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lMHwYXHXCar3"
      },
      "outputs": [],
      "source": [
        "# get data files\n",
        "!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv\n",
        "!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv\n",
        "\n",
        "train_file_path = \"train-data.tsv\"\n",
        "test_file_path = \"valid-data.tsv\""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "g_h508FEClxO"
      },
      "outputs": [],
      "source": [
        "# Función para cargar y preparar datos\n",
        "def load_data(file_path):\n",
        "    df = pd.read_csv(file_path, sep='\\t', header=None, names=['label', 'message'])\n",
        "    df['label'] = df['label'].map({'ham': 0, 'spam': 1})\n",
        "    return df['message'].values, df['label'].values\n",
        "\n",
        "# Cargar datos\n",
        "train_texts, train_labels = load_data(train_file_path)\n",
        "test_texts, test_labels = load_data(test_file_path)\n",
        "\n",
        "# Paso 4: Preprocesamiento de texto\n",
        "# Tokenización: Convertir palabras en números\n",
        "tokenizer = Tokenizer(num_words=10000, oov_token=\"<OOV>\")\n",
        "tokenizer.fit_on_texts(train_texts)\n",
        "\n",
        "# Convertir textos a secuencias numéricas\n",
        "train_sequences = tokenizer.texts_to_sequences(train_texts)\n",
        "test_sequences = tokenizer.texts_to_sequences(test_texts)\n",
        "\n",
        "# Rellenar secuencias para que tengan la misma longitud\n",
        "max_length = 100\n",
        "train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')\n",
        "test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zOMKywn4zReN"
      },
      "outputs": [],
      "source": [
        "# Paso 5: Construir el modelo de clasificación\n",
        "def build_model():\n",
        "    model = Sequential([\n",
        "        # Capa de embedding: convierte índices en vectores densos\n",
        "        Embedding(\n",
        "            input_dim=10000,  # Tamaño del vocabulario\n",
        "            output_dim=128,    # Dimensionalidad del embedding\n",
        "            input_length=max_length  # Longitud de las secuencias\n",
        "        ),\n",
        "\n",
        "        # Capa LSTM bidireccional para capturar contexto\n",
        "        Bidirectional(LSTM(64, return_sequences=True)),\n",
        "        Dropout(0.5),  # Regularización para prevenir sobreajuste\n",
        "\n",
        "        # Segunda capa LSTM\n",
        "        Bidirectional(LSTM(32)),\n",
        "        Dropout(0.5),\n",
        "\n",
        "        # Capa densa final con activación sigmoide\n",
        "        Dense(1, activation='sigmoid')\n",
        "    ])\n",
        "\n",
        "    model.compile(\n",
        "        loss='binary_crossentropy',  # Para clasificación binaria\n",
        "        optimizer=Adam(learning_rate=0.001),\n",
        "        metrics=['accuracy']\n",
        "    )\n",
        "    return model\n",
        "\n",
        "# Crear modelo\n",
        "model = build_model()\n",
        "model.summary()\n",
        "\n",
        "# Paso 6: Entrenar el modelo\n",
        "early_stop = EarlyStopping(\n",
        "    monitor='val_loss',\n",
        "    patience=3,\n",
        "    restore_best_weights=True\n",
        ")\n",
        "\n",
        "history = model.fit(\n",
        "    train_padded,\n",
        "    train_labels,\n",
        "    epochs=20,\n",
        "    validation_split=0.2,\n",
        "    callbacks=[early_stop],\n",
        "    batch_size=64,\n",
        "    verbose=1\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "J9tD9yACG6M9"
      },
      "outputs": [],
      "source": [
        "# function to predict messages based on model\n",
        "# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])\n",
        "\n",
        "# Paso 7: Crear función de predicción\n",
        "def predict_message(pred_text):\n",
        "    # Preprocesar el texto de entrada\n",
        "    sequence = tokenizer.texts_to_sequences([pred_text])\n",
        "    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')\n",
        "\n",
        "    # Predecir probabilidad\n",
        "    prediction_prob = model.predict(padded, verbose=0)[0][0]\n",
        "\n",
        "    # Determinar etiqueta\n",
        "    prediction_label = 'spam' if prediction_prob > 0.5 else 'ham'\n",
        "\n",
        "    return [prediction_prob, prediction_label]\n",
        "\n",
        "# Probar la función\n",
        "pred_text = \"how are you doing today?\"\n",
        "prediction = predict_message(pred_text)\n",
        "print(prediction)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Dxotov85SjsC"
      },
      "outputs": [],
      "source": [
        "# Run this cell to test your function and model. Do not modify contents.\n",
        "def test_predictions():\n",
        "  test_messages = [\"how are you doing today\",\n",
        "                   \"sale today! to stop texts call 98912460324\",\n",
        "                   \"i dont want to go. can we try it a different day? available sat\",\n",
        "                   \"our new mobile video service is live. just install on your phone to start watching.\",\n",
        "                   \"you have won £1000 cash! call to claim your prize.\",\n",
        "                   \"i'll bring it tomorrow. don't forget the milk.\",\n",
        "                   \"wow, is your arm alright. that happened to me one time too\"\n",
        "                  ]\n",
        "\n",
        "  test_answers = [\"ham\", \"spam\", \"ham\", \"spam\", \"spam\", \"ham\", \"ham\"]\n",
        "  passed = True\n",
        "\n",
        "  for msg, ans in zip(test_messages, test_answers):\n",
        "    prediction = predict_message(msg)\n",
        "    if prediction[1] != ans:\n",
        "      passed = False\n",
        "\n",
        "  if passed:\n",
        "    print(\"You passed the challenge. Great job!\")\n",
        "  else:\n",
        "    print(\"You haven't passed yet. Keep trying.\")\n",
        "\n",
        "test_predictions()\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "fcc_sms_text_classification.ipynb",
      "private_outputs": true,
      "provenance": [],
      "toc_visible": true,
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {}
  },
  "nbformat": 4,
  "nbformat_minor": 0
}