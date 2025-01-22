import os
import json
from datetime import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from src.MedMnist.conponents.custom_cnn import experiment_models
from src.MedMnist.logging.logger import logger
from src.MedMnist.pipeline.train import optimize_one
from src.MedMnist.entity.dataset import load_data
from src.MedMnist.utils.plot import plot_training_history
from src.MedMnist.logging.neptune import NeptuneLogger



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

    optimizer_dict = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "adamax": tf.keras.optimizers.Adamax
    }

    # Konfiguracja modelu
    model.compile(
    optimizer=optimizer_dict[best_params['optimizer']](learning_rate=best_params["learning_rate"]),
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





  
    