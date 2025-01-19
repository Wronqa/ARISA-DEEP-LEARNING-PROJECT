import optuna
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Nowe API Neptune (bez 'contrib')
# Zakładam, że w innym miejscu masz: run = neptune.init_run(...)
# i przekazujesz to 'run' jako argument.
# import neptune

from src.MedMnist.config.configuration import optymized_params
from src.MedMnist.conponents.custom_cnn import experiment_models


# -----------------------------------------------------------------------------
# CALLBACK 1: Keras -> Neptune epoka-po-epoce
# -----------------------------------------------------------------------------
class NeptuneKerasEpochLogger(tf.keras.callbacks.Callback):
    """
    Callback logujący do Neptune train/val loss i accuracy 
    na koniec każdej epoki.
    """

    def __init__(self, run, model_name, trial_id):
        """
        :param run: obiekt Neptune Run (np. przekazany z maina)
        :param model_name: nazwa modelu lub wersja (np. 'modelA')
        :param trial_id: numer próby w Optunie (trial.number)
        """
        super().__init__()
        self.run = run
        self.model_name = model_name
        self.trial_id = trial_id

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # np. "optuna/modelA/trial_5/train/loss", "optuna/modelA/trial_5/val/loss", itp.
        base_ns = f"optuna/{self.model_name}/trial_{self.trial_id}"

        # Logs z train:
        if "loss" in logs:
            self.run[f"{base_ns}/train/loss"].append(logs["loss"])
        if "accuracy" in logs:
            self.run[f"{base_ns}/train/accuracy"].append(logs["accuracy"])

        # Logs z val:
        if "val_loss" in logs:
            self.run[f"{base_ns}/val/loss"].append(logs["val_loss"])
        if "val_accuracy" in logs:
            self.run[f"{base_ns}/val/accuracy"].append(logs["val_accuracy"])


# -----------------------------------------------------------------------------
# CALLBACK 2: Optuna -> Neptune (logowanie param/ value)
# -----------------------------------------------------------------------------
class NeptuneOptunaCallback:
    """
    Prosty callback do integracji Optuny z nowym Neptune (API z init_run()).
    Po każdej próbie loguje do run['optuna/<model_key>/trial_<trial_number>/...']:
      - parametry próby (params/...)
      - wartość funkcji celu (value)
    Dodatkowo, jeśli bieżąca próba jest najlepsza, aktualizuje best_value i best_trial_number.
    """

    def __init__(self, run, base_namespace="optuna", model_key=None):
        """
        :param run: obiekt Neptune Run
        :param base_namespace: ścieżka w drzewie logów Neptune, np. "optuna"
        :param model_key: nazwa modelu (np. "modelA") – będzie częścią ścieżki
        """
        self.run = run
        self.base_namespace = base_namespace
        self.model_key = model_key  # np. "modelA" / "modelB"

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        # Zbuduj ścieżkę np. "optuna/modelA/trial_0"
        if self.model_key is not None:
            trial_ns = f"{self.base_namespace}/{self.model_key}/trial_{trial.number}"
            best_ns = f"{self.base_namespace}/{self.model_key}"
        else:
            trial_ns = f"{self.base_namespace}/trial_{trial.number}"
            best_ns = f"{self.base_namespace}"

        # Loguj parametry próby
        for param_name, param_value in trial.params.items():
            self.run[f"{trial_ns}/params/{param_name}"] = param_value

        # Loguj wartość funkcji celu
        if trial.value is not None:
            self.run[f"{trial_ns}/value"] = trial.value

        # Jeśli ta próba jest najlepsza:
        if study.best_trial == trial:
            self.run[f"{best_ns}/best_value"] = study.best_value
            self.run[f"{best_ns}/best_trial_number"] = trial.number


# -----------------------------------------------------------------------------
# FUNKCJA CELU (objective) - trenowanie modelu, zwracanie maks. val_accuracy
# -----------------------------------------------------------------------------
def objective(trial, x_train, y_train, x_test, y_test, num_classes, model_version, run):
    """
    Funkcja celu Optuny. Loguje do Neptune przebieg każdej epoki 
    (NeptuneKerasEpochLogger) oraz finalną wartość (maksymalną val_accuracy).
    """
    params = {
        "learning_rate": trial.suggest_float("learning_rate", *optymized_params["learning_rate_range"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", optymized_params["batch_size_options"]),
        "dropout_rate": trial.suggest_float("dropout_rate", *optymized_params["dropout_rate_range"]),
        "filters_0": trial.suggest_categorical("filters_0", optymized_params["filters_0_options"]),
        "filters_1": trial.suggest_categorical("filters_1", optymized_params["filters_1_options"]),
        "filters_2": trial.suggest_categorical("filters_2", optymized_params["filters_2_options"]),
        "filters_3": trial.suggest_categorical("filters_3", optymized_params["filters_3_options"]),
        "dense_units": trial.suggest_categorical("dense_units", optymized_params["dense_units_options"]),
        "activation_function": trial.suggest_categorical("activation_function", optymized_params["activation_functions"]),
        "optimizer": trial.suggest_categorical("optimizer", optymized_params["optimizer_options"]),
        "epochs": trial.suggest_categorical("epochs", optymized_params["epochs_range"]),
        "early_stopping_patience": trial.suggest_int("early_stopping_patience", *optymized_params["early_stopping_patience"]),
    }

    # Tworzymy model
    model_function = experiment_models[f"{model_version}"]
    model = model_function((28, 28, 1), num_classes, params)

    # Optymalizator
    optimizer_dict = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "adamax": tf.keras.optimizers.Adamax
    }
    optimizer = optimizer_dict[params["optimizer"]](learning_rate=params["learning_rate"])

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacki
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=params["early_stopping_patience"],
        restore_best_weights=True
    )

    neptune_keras_cb = NeptuneKerasEpochLogger(
        run=run,
        model_name=model_version,
        trial_id=trial.number
    )

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

    # Trenowanie
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=params["batch_size"]),
        validation_data=(x_test, y_test),
        epochs=params["epochs"],
        verbose=0,
        callbacks=[early_stopping, neptune_keras_cb]
    )

    # Maks. dokładność walidacyjna
    val_accuracy = max(history.history['val_accuracy'])

    K.clear_session()
    return val_accuracy


def optimize_one(x_train, y_train, x_test, y_test, num_classes, run, model):
    """
    - x_train, y_train, x_test, y_test, num_classes: dane i info o klasach
    - run: obiekt Neptune (init_run), do którego chcemy logować
    - model: klucz modelu, dla którego chcemy optymalizować hiperparametry
    """
    study = optuna.create_study(direction="maximize")

    # Nasz wlasny callback do logowania param/wartości w Neptune
    neptune_optuna_callback = NeptuneOptunaCallback(
        run=run,
        base_namespace="optuna",
        model_key=model
    )

    # Wywołujemy study.optimize, przekazując objective + callback
    study.optimize(
        lambda trial: objective(
            trial=trial,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            model_version=model,
            run=run
        ),
        n_trials=20,
        callbacks=[neptune_optuna_callback]
    )

    # Zbieramy najlepsze parametry
    best_results = study.best_params
    print(f"Best parameters for {model}: {study.best_params}")

    # Możesz dodatkowo logować końcowe rezultaty do Neptune
    run[f"optuna/{model}/best_value"] = study.best_value
    run[f"optuna/{model}/best_params"] = study.best_params

    return best_results
    
    


# -----------------------------------------------------------------------------
# GŁÓWNA FUNKCJA: OPTYMALIZACJA HIPERPARAMETRÓW DLA RÓŻNYCH MODELI
# -----------------------------------------------------------------------------
def optimize_hyperparameters(x_train, y_train, x_test, y_test, num_classes, run):
    """
    - x_train, y_train, x_test, y_test, num_classes: dane i info o klasach
    - run: obiekt Neptune (init_run), do którego chcemy logować
    """
    best_results = {}
    model_keys = list(experiment_models.keys())

    for key in model_keys:
        print(f"Optimizing hyperparameters for model version: {key}")

        best_results[key] = optimize_one(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            run=run,
            model=key
        )

    return best_results