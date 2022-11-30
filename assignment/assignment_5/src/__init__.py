import numpy as np 
import tensorflow.keras as KERAS 
import tensorflow as tf
import matplotlib.pyplot as plt 


def label_shrink(input_label, pre_map = {}):
    cat_idx = 0
    mapping = pre_map
    res = [] 

    for l in input_label:
        
        l = l[0]

        if l not in mapping:
            mapping[l] = cat_idx
            cat_idx += 1

        new_label = mapping[l]

        res.append(new_label)
    
    res = np.asarray(res)
    res = np.reshape(res, (-1, 1))
    return res, mapping


def into_three_ch(data, size):
    res = []
    for img_array in data:
        tensor = tf.convert_to_tensor(img_array)
        tensor = tf.image.resize(tensor, size)
        tensor = tf.image.grayscale_to_rgb(tensor)
        res.append(tensor)
    return np.asarray(res)



def plot_history(history):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    dummy_x = list(range(len(acc)))

    plt.plot(dummy_x, acc)
    plt.plot(dummy_x, val_acc)
    plt.legend(['acc', 'val_acc'])

    print('categorical_accuracy')
    print(history.history['categorical_accuracy'][-1])
    print('val_categorical_accuracy')
    print(history.history['val_categorical_accuracy'][-1])

    plt.show()

def plot_history_rnn(history):
    acc = history.history['mean_squared_error']
    val_acc = history.history['val_mean_squared_error']
    dummy_x = list(range(len(acc)))

    plt.plot(dummy_x, acc)
    plt.plot(dummy_x, val_acc)
    plt.legend(['mean_squared_error', 'val_mean_squared_error'])

    plt.show()

    print('mean_squared_error')
    print(history.history['mean_squared_error'][-1])
    print('val_mean_squared_error')
    print(history.history['val_mean_squared_error'][-1])

def compile_fit_rnn(model, train, test):
    x_train, y_train = train
    x_test, y_test = test  

    model.compile(
        loss = KERAS.losses.MeanSquaredError(), 
        optimizer=KERAS.optimizers.RMSprop(), 
        metrics=[KERAS.metrics.MeanSquaredError()]
    )

    model.summary()

    history = model.fit(
        x_train, y_train, 
        epochs = 100, 
        validation_data=(x_test, y_test), 
        verbose=0
    )

    return history

def sampling_rnn(data_array, lookback):
    x, y = [], [] 
    for i in range(data_array.shape[0] - lookback):
        
        sample = data_array[i:i + lookback, 0]
        target = data_array[i + lookback, 0]

        x.append(sample)
        y.append(target)

    x = np.asarray(x)
    y = np.asarray(y)

    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)

    return x, y      