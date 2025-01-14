from tensorflow.keras import layers, models

def create_model(input_shape, num_classes, params):
    model = models.Sequential()
    model.add(layers.Conv2D(params["filters_1"], (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(params["filters_2"], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(params["dense_units"], activation='relu'))
    model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model