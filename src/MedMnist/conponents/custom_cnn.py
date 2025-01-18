from tensorflow.keras import layers, models

def create_model(input_shape, num_classes, params):
    model = models.Sequential()
    model.add(layers.Conv2D(params["filters_1"], (3, 3), 
    activation='relu', 
    input_shape=input_shape
    ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(params["filters_2"], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(params["dense_units"], activation='relu'))
    model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

#Wersja 1: Zwiększona liczba filtrów i dodatkowa warstwa Conv2D

def create_model_v1(input_shape, num_classes, params):
    model = models.Sequential()
    model.add(layers.Conv2D(params["filters_1"], (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(params["filters_2"], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(params["filters_3"], (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(params["dense_units"], activation='relu'))
    model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

#Wersja 2: Użycie Batch Normalization dla stabilności uczenia
def create_model_v2(input_shape, num_classes, params):
    model = models.Sequential()
    model.add(layers.Conv2D(params["filters_1"], (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(params["filters_2"], (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(params["dense_units"], activation='relu'))
    model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

#Wersja 3: Głębsza sieć z dodatkowymi Dense i Dropout
def create_model_v3(input_shape, num_classes, params):
    model = models.Sequential()
    model.add(layers.Conv2D(params["filters_1"], (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(params["filters_2"], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(params["dense_units"], activation='relu'))
    model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(params["dense_units"] // 2, activation='relu'))
    model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


#Wersja 4: Użycie GlobalAveragePooling2D zamiast Flatten
def create_model_v4(input_shape, num_classes, params):
    model = models.Sequential()
    model.add(layers.Conv2D(params["filters_1"], (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(params["filters_2"], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(params["dense_units"], activation='relu'))
    model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

#Wersja 5: Dodanie warstwy Conv2D z większym kernelem (5x5)
def create_model_v5(input_shape, num_classes, params):
    model = models.Sequential()
    model.add(layers.Conv2D(params["filters_1"], (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(params["filters_2"], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(params["dense_units"], activation='relu'))
    model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


experiment_models = {
    "model_v1": create_model_v1,
    "model_v2": create_model_v2,
    "model_v3": create_model_v3,
    "model_v4": create_model_v4,
    "model_v5": create_model_v5,
}