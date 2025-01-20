from tensorflow.keras import layers, models, regularizers

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

def create_model_v6(input_shape, num_classes, params):
    model = models.Sequential()
    
    # Pierwsza warstwa konwolucyjna 5x5
    model.add(layers.Conv2D(params["filters_0"], (5, 5), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Druga warstwa konwolucyjna 3x3
    model.add(layers.Conv2D(params["filters_1"], (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Trzecia warstwa konwolucyjna 3x3
    model.add(layers.Conv2D(params["filters_2"], (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Czwarta warstwa konwolucyjna 3x3 (lub 1x1 dla eksperymentów)
    model.add(layers.Conv2D(params["filters_3"], (3, 3), activation='relu', padding='same'))
    
    # Spłaszczenie wyników
    model.add(layers.Flatten())
    
    # Warstwa gęsta
    model.add(layers.Dense(params["dense_units"], activation='relu'))
    model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def create_model_v7(input_shape, num_classes, params):
    model = models.Sequential()
    # Dodanie dwóch warstw konwolucyjnych na początek
    model.add(layers.Conv2D(params["filters_0"], (3, 3), activation='relu', input_shape=input_shape))  # Wymiary: (H-2, W-2, filters_0)
    model.add(layers.Conv2D(params["filters_1"], (3, 3), activation='relu'))  # Wymiary: (H-4, W-4, filters_1)
    model.add(layers.MaxPooling2D((2, 2)))  # Wymiary: (H//2-2, W//2-2, filters_1)
    # Pozostała część modelu
    model.add(layers.Conv2D(params["filters_2"], (3, 3), activation='relu'))  # Wymiary zmniejszają się
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(params["filters_3"], (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(params["dense_units"], activation='relu'))
    model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def create_model_v8(input_shape, num_classes, params):
    model = models.Sequential()
    
    # Blok 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))  # Redukcja do 14x14
    
    # Blok 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))  # Redukcja do 7x7
    
    # Blok 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))  # Redukcja do 3x3
    
    # Blok 4 (dodatkowe warstwy konwolucyjne)
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    
    # Spłaszczenie i warstwy gęste
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))  # Większa gęsta warstwa
    model.add(layers.Dropout(0.5))  # Dropout dla regularyzacji
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def create_model_enhanced(input_shape, num_classes, params):
    model = models.Sequential()

    # Blok 1
    model.add(layers.Conv2D(params["filters_0"], (3, 3), padding='same',
                            input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(params["l2_reg"])))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(params["filters_1"], (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(params["l2_reg"])))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(params["dropout_rate"]))

    # Blok 2
    model.add(layers.Conv2D(params["filters_2"], (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(params["l2_reg"])))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(params["filters_3"], (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(params["l2_reg"])))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(params["dropout_rate"]))

    # Blok 3
    model.add(layers.Conv2D(params["filters_4"], (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(params["l2_reg"])))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(params["dropout_rate"]))

    # Global Average Pooling zamiast Flatten
    model.add(layers.GlobalAveragePooling2D())

    # Warstwa gęsta + wyjście
    model.add(layers.Dense(params["dense_units"], activation='relu'))
    model.add(layers.Dropout(params["dropout_dense"]))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

experiment_models = {
    "model_v1": create_model_v1,
    "model_v2": create_model_v2,
    "model_v3": create_model_v3,
    "model_v4": create_model_v4,
    "model_v5": create_model_v5,
    "model_v6": create_model_v6,
    "model_v7": create_model_v7,
    'model_v8': create_model_v8,
    "model_v9": create_model_enhanced
}