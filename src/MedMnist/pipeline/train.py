import optuna
from tensorflow.keras import backend as K
from src.MedMnist.conponents.model import create_model
import tensorflow as tf

def objective(trial, x_train, y_train, x_test, y_test, num_classes):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "filters_1": trial.suggest_categorical("filters_1", [16, 32, 64]),
        "filters_2": trial.suggest_categorical("filters_2", [64, 128, 256]),
        "dense_units": trial.suggest_categorical("dense_units", [64, 128, 256]),
    }

    model = create_model((28, 28, 1), num_classes, params)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=params["batch_size"],
        verbose=0
    )
    val_accuracy = max(history.history['val_accuracy'])
    K.clear_session()
    return val_accuracy

def optimize_hyperparameters(x_train, y_train, x_test, y_test, num_classes):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, num_classes), n_trials=20)
    return study.best_paramslo