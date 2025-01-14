from medmnist import OrganAMNIST
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_data():
    train_dataset = OrganAMNIST(split='train', download=True)
    test_dataset = OrganAMNIST(split='test', download=True)
    x_train, y_train = train_dataset.imgs, train_dataset.labels
    x_test, y_test = test_dataset.imgs, test_dataset.labels

    # Normalizacja i reshaping
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # One-hot encoding
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return x_train, y_train, x_test, y_test, num_classes