import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.MedMnist.conponents.custom_cnn import experiment_models
from src.MedMnist.logging.logger import logger
from src.MedMnist.pipeline.train import optimize_one
from src.MedMnist.entity.dataset import load_data
from src.MedMnist.utils.plot import plot_training_history
from src.MedMnist.logging.neptune import NeptuneLogger


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
    best_params = optimize_one(x_train, y_train, x_test, y_test, num_classes, run, choosen_model)
    logger.info(f"Training chosen model version: {choosen_model}...")

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
