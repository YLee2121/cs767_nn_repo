import tensorflow.keras as keras

def model_activation(activation_func):

    model = keras.models.Sequential()

    model.add( 
        keras.layers.Dense(16, activation=activation_func, input_shape=(2000,))
    )
    model.add(
        keras.layers.Dense(16, activation=activation_func)
    )
    model.add(
        keras.layers.Dense(1, activation='sigmoid')
    )

    return model