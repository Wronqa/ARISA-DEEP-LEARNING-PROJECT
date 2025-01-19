import argparse
import os
import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Zwróć uwagę, że korzystamy ze "starego" importu neptune,
# a nie `import neptune.new`.
import neptune

from src.MedMnist.entity.dataset import load_data
from src.MedMnist.pipeline.train import optimize_hyperparameters, optimize_one
from src.MedMnist.conponents.custom_cnn import experiment_models
from src.MedMnist.logging.logger import setup_logger
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# CALLBACK DO LOGOWANIA METRYK EPOKA-PO-EPOCE W NEPTUNE
# -----------------------------------------------------------------------------
class NeptuneLogger(tf.keras.callbacks.Callback):
    """
    Callback do logowania metryk trenowania i walidacji do Neptune
    po każdej epoce.
    """
    def __init__(self, run, prefix="model"):
        super().__init__()
        self.run = run
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # logowanie metryk trenowania (jeśli istnieją w logs)
        if 'loss' in logs:
            self.run[f"{self.prefix}/train/loss"].append(logs['loss'])
        if 'accuracy' in logs:
            self.run[f"{self.prefix}/train/accuracy"].append(logs['accuracy'])

        # logowanie metryk walidacyjnych (jeśli istnieją w logs)
        if 'val_loss' in logs:
            self.run[f"{self.prefix}/val/loss"].append(logs['val_loss'])
        if 'val_accuracy' in logs:
            self.run[f"{self.prefix}/val/accuracy"].append(logs['val_accuracy'])


def train_all_models(experiment_name, experiment_dir, run):
    """
    Trenuje wszystkie zdefiniowane modele (experiment_models) z automatyczną 
    optymalizacją hiperparametrów, loguje wyniki do Neptune, 
    zapisuje model i wykresy lokalnie.
    """
    x_train, y_train, x_test, y_test, num_classes = load_data()

    logger.info("Optimizing hyperparameters...")
    best_params_all_models = optimize_hyperparameters(x_train, y_train, x_test, y_test, num_classes, run)

    for version, create_model in experiment_models.items():
        logger.info(f"Training model version: {version}...")

        # Określamy przestrzeń nazw dla logów w Neptune (np. "model_modelA")
        model_ns = f"model_{version}"

        # Pobierz najlepsze hiperparametry
        best_params = best_params_all_models[version]

        # Logowanie hiperparametrów do Neptune
        run[f"{model_ns}/hyperparameters"] = best_params

        # Tworzenie i kompilacja modelu
        model = create_model((28, 28, 1), num_classes, best_params)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Przygotowanie callbacków
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        neptune_cb = NeptuneLogger(run, prefix=model_ns)

        # Augmentacja
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(x_train)

        # Trening
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=best_params["batch_size"]),
            validation_data=(x_test, y_test),
            epochs=50,
            verbose=2,
            callbacks=[early_stopping, neptune_cb]
        )

        # Ewaluacja
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        logger.info(f"Test accuracy for model {version}: {test_accuracy:.4f}")

        # Logowanie wyników testu do Neptune
        run[f"{model_ns}/test/loss"] = test_loss
        run[f"{model_ns}/test/accuracy"] = test_accuracy

        # Zapis modelu lokalnie
        model_dir = f"{experiment_dir}/models/model_{version}"
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, "model.h5"))
        logger.info(f"Model {version} saved to {model_dir}/model.h5")

        # Zapis hiperparametrów lokalnie
        params_file = os.path.join(model_dir, "hyperparameters.json")
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Hyperparameters for model {version} saved to {params_file}")

        # Wykres i upload do Neptune
        plot_training_history(history, version, experiment_dir, run)


def train_choosen_model(experiment_name, experiment_dir, choosen_model, run):
    """
    Trenuje wybrany model (choosen_model) z automatyczną optymalizacją 
    hiperparametrów, loguje wyniki do Neptune, zapisuje model i wykresy lokalnie.
    """
    if choosen_model not in experiment_models:
        logger.error(f"Model version '{choosen_model}' not found. Available models: {list(experiment_models.keys())}")
        return

    x_train, y_train, x_test, y_test, num_classes = load_data()

    logger.info("Optimizing hyperparameters for the chosen model...")
    best_params_all_models = optimize_one(x_train, y_train, x_test, y_test, num_classes, run, choosen_model)
    logger.info(f"Training chosen model version: {choosen_model}...")

    model_ns = f"model_{choosen_model}"
    best_params = best_params_all_models[choosen_model]
    run[f"{model_ns}/hyperparameters"] = best_params

    create_model = experiment_models[choosen_model]
    model = create_model((28, 28, 1), num_classes, best_params)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    neptune_cb = NeptuneLogger(run, prefix=model_ns)

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=best_params["batch_size"]),
        validation_data=(x_test, y_test),
        epochs=best_params["epochs"],
        verbose=2,
        callbacks=[early_stopping, neptune_cb]
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Test accuracy for model {choosen_model}: {test_accuracy:.4f}")
    run[f"{model_ns}/test/loss"] = test_loss
    run[f"{model_ns}/test/accuracy"] = test_accuracy

    model_dir = f"{experiment_dir}/models/model_{choosen_model}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.h5"))
    logger.info(f"Model {choosen_model} saved to {model_dir}/model.h5")

    params_file = os.path.join(model_dir, "hyperparameters.json")
    with open(params_file, "w") as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Hyperparameters for model {choosen_model} saved to {params_file}")

    plot_training_history(history, choosen_model, experiment_dir, run)


def train_single_model(experiment_name, experiment_dir, choosen_model, run):
    """
    Trenuje pojedynczy model (choosen_model) na podstawie już istniejących 
    hiperparametrów (załadowanych z pliku 'hyperparameters.json').
    """
    x_train, y_train, x_test, y_test, num_classes = load_data()

    model_dir = f"{experiment_dir}/models/{choosen_model}"
    params_file = os.path.join(model_dir, "hyperparameters.json")

    if not os.path.exists(params_file):
        logger.error(f"Hyperparameters file not found for model {choosen_model} at {params_file}")
        return

    with open(params_file, "r") as f:
        best_params = json.load(f)

    logger.info(f"Loaded hyperparameters for model {choosen_model}: {best_params}")

    model_ns = f"model_{choosen_model}"
    run[f"{model_ns}/hyperparameters"] = best_params

    create_model = experiment_models[choosen_model]
    model = create_model((28, 28, 1), num_classes, best_params)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    neptune_cb = NeptuneLogger(run, prefix=model_ns)

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=best_params["batch_size"]),
        validation_data=(x_test, y_test),
        epochs=50,
        verbose=2,
        callbacks=[early_stopping, neptune_cb]
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Test accuracy for model {choosen_model}: {test_accuracy:.4f}")
    run[f"{model_ns}/test/loss"] = test_loss
    run[f"{model_ns}/test/accuracy"] = test_accuracy

    model_dir = f"{experiment_dir}/models/model_{choosen_model}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.h5"))
    logger.info(f"Model {choosen_model} saved to {model_dir}/model.h5")

    plot_training_history(history, choosen_model, experiment_dir, run)


def fine_tune_model(experiment_name, experiment_dir, choosen_model, fine_tune_params_path, run):
    """
    Fine-tuning istniejącego modelu (choosen_model) używając hiperparametrów 
    z pliku JSON (fine_tune_params_path). Logujemy także przebieg do Neptune.
    """
    x_train, y_train, x_test, y_test, num_classes = load_data()

    # Ścieżki do modelu
    model_dir = f"{experiment_dir}/models/{choosen_model}"
    pretrained_model_file = os.path.join(model_dir, "model.h5")

    if not os.path.exists(pretrained_model_file):
        logger.error(f"Pretrained model not found for model {choosen_model} at {pretrained_model_file}")
        return

    if fine_tune_params_path is None or not os.path.exists(fine_tune_params_path):
        logger.error(f"No valid fine-tune params path provided. Provided path: {fine_tune_params_path}")
        return

    with open(fine_tune_params_path, "r") as f:
        best_params = json.load(f)
    logger.info(f"Loaded fine-tune parameters from {fine_tune_params_path}")

    required_keys = ["learning_rate", "batch_size", "optimizer", "epochs", "early_stopping_patience"]
    missing_keys = [key for key in required_keys if key not in best_params]
    if missing_keys:
        logger.error(f"Missing required keys in fine-tune params: {missing_keys}")
        return

    model = load_model(pretrained_model_file)

    # Logowanie hiperparametrów do Neptune
    model_ns = f"model_{choosen_model}_fine_tune"
    run[f"{model_ns}/hyperparameters"] = best_params

    # Konfiguracja modelu
    model.compile(
        optimizer=tf.keras.optimizers.get(best_params["optimizer"])(
            learning_rate=best_params["learning_rate"]
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=best_params["early_stopping_patience"],
        restore_best_weights=True
    )
    neptune_cb = NeptuneLogger(run, prefix=model_ns)

    # Augmentacja
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=best_params["batch_size"]),
        validation_data=(x_test, y_test),
        epochs=best_params["epochs"],
        verbose=2,
        callbacks=[early_stopping, neptune_cb]
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Test accuracy for fine-tuned model {choosen_model}: {test_accuracy:.4f}")
    run[f"{model_ns}/test/loss"] = test_loss
    run[f"{model_ns}/test/accuracy"] = test_accuracy

    unique_id = time.strftime("%Y%m%d_%H%M%S")  
    fine_tuned_dir = os.path.join(model_dir, f"fine_tuned_{unique_id}")
    os.makedirs(fine_tuned_dir, exist_ok=True)

    fine_tuned_model_file = os.path.join(fine_tuned_dir, "fine_tuned_model.h5")
    model.save(fine_tuned_model_file)
    logger.info(f"Fine-tuned model saved to {fine_tuned_model_file}")

    fine_tuned_params_file = os.path.join(fine_tuned_dir, "fine_tune_params.json")
    with open(fine_tuned_params_file, "w") as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Fine-tuned parameters saved to {fine_tuned_params_file}")

    plot_training_history(history, f"{choosen_model}_fine_tuned", fine_tuned_dir, run)


def plot_training_history(history, version, experiment_dir, run):
    """
    Rysuje wykresy treningu (accuracy i loss), zapisuje je lokalnie
    oraz jako artefakt w Neptune.
    """
    plt.figure(figsize=(10, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title(f'Model {version} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title(f'Model {version} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_path = f"{experiment_dir}/plots/model_{version}_training_results.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    # Wysyłamy wykres do Neptune jako artefakt
    run[f"model_{version}/plots/training_results"].upload(plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate models.")
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--mode", type=str, default='train_all', help="Mode: train_all, train_single, fine_tune...")
    parser.add_argument("--choosen-model", type=str, help="Specify a model version to train/fine_tune")
    parser.add_argument("--fine-tune-params-path", type=str, help="Path to the fine-tune parameters JSON file.")
    args = parser.parse_args()

    experiment_name = args.experiment_name
    choosen_model = args.choosen_model
    mode = args.mode
    fine_tune_params_path = args.fine_tune_params_path

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = f"experiments/{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting pipeline for experiment: %s at %s", experiment_name, current_date)
    # Wczytaj zmienne środowiskowe z pliku .env
    load_dotenv()
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    if not api_token:
        raise ValueError("NEPTUNE_API_TOKEN not found in environment variables.")

    run = neptune.init_run(
    project="pgawzynski.backup/ArisaDeepLearning",
    api_token=api_token,
    )
    
    run["sys/name"] = experiment_name  # nazwa runa

    try:
        if mode == "train_single":
            logger.info("Training single model: %s", choosen_model)
            train_choosen_model(experiment_name, experiment_dir, choosen_model, run)

        elif mode == "train_all":
            logger.info("Training all models.")
            train_all_models(experiment_name, experiment_dir, run)

        elif mode == "fine_tune":
            logger.info("Fine-tuning model: %s", choosen_model)
            fine_tune_model(experiment_name, experiment_dir, choosen_model, fine_tune_params_path, run)

        else:
            logger.error(f"Unknown mode: {mode}. Available modes: train_all, train_single, fine_tune.")

    finally:
        # Po zakończeniu (lub błędzie) zamykamy run w Neptune.
        run.stop()

    logger.info("Pipeline completed.")