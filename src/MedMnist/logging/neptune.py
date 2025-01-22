
import tensorflow as tf


class NeptuneLogger(tf.keras.callbacks.Callback):
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
