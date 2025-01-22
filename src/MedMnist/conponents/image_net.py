import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.applications import (
    VGG16, ResNet50, InceptionV3, MobileNetV2
)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import neptune

def use_image_net(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    run,                              
    base_model_name: str = "InceptionV3", 
    input_shape: tuple = (224, 224, 3),
    freeze_until: int = 150,  
    hidden_units: int = 256,  
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-4,
    batch_size: int = 64,  
    epochs: int = 50,  
    augment_data: bool = True
    ):

    
    if base_model_name == "VGG16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif base_model_name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif base_model_name == "InceptionV3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif base_model_name == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    else:
        raise ValueError(f"Unsuported base_model_name={base_model_name}")

   

    def process_image(image, label):

        image = tf.cast(image, tf.float32)
        if tf.shape(image)[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, [input_shape[0], input_shape[1]])

        if augment_data:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

        image = preprocess_input(image)
        return image, label

 

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = (train_dataset
                     .shuffle(10000)
                     .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    test_dataset = (test_dataset
                    .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))


    if base_model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == "InceptionV3":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == "MobileNetV2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True

 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if hidden_units > 0:
        x = Dense(hidden_units, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(1e-4))(x)
    model = Model(inputs=base_model.input, outputs=outputs)


    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )



    # ----------------------------------------------------------------------------
    class NeptuneLogger(tf.keras.callbacks.Callback):

        def __init__(self, run, prefix="training"):
            super().__init__()
            self.run = run
            self.prefix = prefix

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            for metric, value in logs.items():
                self.run[f"{self.prefix}/{metric}"].log(value)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    neptune_logger = NeptuneLogger(run)

 

    run["parameters"] = {
        "base_model": base_model_name,
        "input_shape": input_shape,
        "freeze_until": freeze_until,
        "hidden_units": hidden_units,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "augment_data": augment_data
    }

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[early_stopping, neptune_logger],
        verbose=2
    )

    
    eval_results = model.evaluate(test_dataset, verbose=0, return_dict=True)
    for metric, value in eval_results.items():
        run[f"evaluation/{metric}"] = value

    print(f"[INFO] Test Metrics: {eval_results}")


    model.save("model.h5")
    run["model/saved_model"].upload("model.h5")

    return model, history, eval_results["loss"], eval_results["accuracy"]