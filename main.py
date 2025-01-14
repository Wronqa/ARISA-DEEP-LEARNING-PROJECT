from src.MedMnist.entity.dataset import load_data
from src.MedMnist.pipeline.train import optimize_hyperparameters
from src.MedMnist.conponents.model import create_model
from src.MedMnist.logging.logger import setup_logger
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf  # Import TensorFlow
import os


logger = setup_logger()

if __name__ == "__main__":
    x_train, y_train, x_test, y_test, num_classes = load_data()

    logger.info("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(x_train, y_train, x_test, y_test, num_classes)

    logger.info(f"Best parameters: {best_params}")

    logger.info("Training final model with cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    val_accuracies = []

    for train_idx, val_idx in kfold.split(x_train):
        x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model = create_model((28, 28, 1), num_classes, best_params)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            x_train_fold, y_train_fold,
            validation_data=(x_val_fold, y_val_fold),
            epochs=5,
            batch_size=best_params["batch_size"],
            verbose=2
        )

        val_accuracy = max(history.history['val_accuracy'])
        val_accuracies.append(val_accuracy)

    logger.info(f"Average validation accuracy: {np.mean(val_accuracies):.4f}")