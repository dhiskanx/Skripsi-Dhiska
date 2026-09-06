# Data Loader
import os
import pickle
import tqdm
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam
import time
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation, \
    Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class DataLoader(object):
    def __init__(self, width, height, cells, data_path, output_path):
        self.width = width
        self.height = height
        self.cells = cells
        self.data_path = data_path
        self.output_path = output_path

    def _open_image(self, path):
        image = Image.open(path).convert('L')
        image = image.resize((self.width, self.height))
        data = np.asarray(image, dtype='float64')
        return data

    def convert_image_to_array(self, person, image_num, data_path, predict=False):
        max_zeros = 4
        image_num = '0' * max_zeros + image_num
        image_num = image_num[-max_zeros:]

        image_path = os.path.join(
            data_path, 'lfw2', person, f'{person}_{image_num}.jpg'
        )

        image_data = self._open_image(image_path)

        if not predict:
            image_data = image_data.reshape(self.width, self.height, self.cells)

        return image_data

    def load(self, set_name):
        file_path = os.path.join(self.data_path, f'{set_name}.txt')
        print(file_path)
        print('Loading dataset...')

        x_first, x_second, y, names = [], [], [], []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in tqdm.tqdm(lines):
            line = line.split()

            if len(line) == 4:
                names.append(line)
                p1, i1, p2, i2 = line
                x_first.append(self.convert_image_to_array(p1, i1, self.data_path))
                x_second.append(self.convert_image_to_array(p2, i2, self.data_path))
                y.append(0)

            elif len(line) == 3:
                names.append(line)
                p, i1, i2 = line
                x_first.append(self.convert_image_to_array(p, i1, self.data_path))
                x_second.append(self.convert_image_to_array(p, i2, self.data_path))
                y.append(1)

        print('Done loading dataset')

        with open(self.output_path, 'wb') as f:
            pickle.dump([[x_first, x_second], y, names], f)


print("Loaded data loader")


# Siamese Neural Network
class SiameseNetwork(object):
    def __init__(self, seed, width, height, cells, loss, metrics, optimizer, dropout_rate):
        K.clear_session()
        self.load_file = None
        self.seed = seed
        self.initialize_seed()
        self.optimizer = optimizer

        input_shape = (width, height, cells)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        model = self._get_architecture(input_shape)
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_siamese_dist = L1_layer([encoded_l, encoded_r])
        L1_siamese_dist = Dropout(dropout_rate)(L1_siamese_dist)

        prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(L1_siamese_dist)

        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
        self.siamese_net = siamese_net
        self.siamese_net.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def initialize_seed(self):
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def initialize_weights(self, shape, dtype=None):
        return K.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype, seed=self.seed)

    def initialize_bias(self, shape, dtype=None):
        return K.random_normal(shape, mean=0.5, stddev=0.01, dtype=dtype, seed=self.seed)

    def _get_architecture(self, input_shape):
        model = Sequential()
        model.add(
            Conv2D(filters=64,
                   kernel_size=(10, 10),
                   input_shape=input_shape,
                   kernel_initializer=self.initialize_weights,
                   kernel_regularizer=l2(2e-4),
                   name='Conv1'
                   ))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D())

        model.add(
            Conv2D(filters=128,
                   kernel_size=(7, 7),
                   kernel_initializer=self.initialize_weights,
                   bias_initializer=self.initialize_bias,
                   kernel_regularizer=l2(2e-4),
                   name='Conv2'
                   ))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D())

        model.add(
            Conv2D(filters=128,
                   kernel_size=(4, 4),
                   kernel_initializer=self.initialize_weights,
                   bias_initializer=self.initialize_bias,
                   kernel_regularizer=l2(2e-4),
                   name='Conv3'
                   ))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D())

        model.add(
            Conv2D(filters=256,
                   kernel_size=(4, 4),
                   kernel_initializer=self.initialize_weights,
                   bias_initializer=self.initialize_bias,
                   kernel_regularizer=l2(2e-4),
                   name='Conv4'
                   ))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(
            Dense(4096,
                  activation='sigmoid',
                  kernel_initializer=self.initialize_weights,
                  kernel_regularizer=l2(2e-3),
                  bias_initializer=self.initialize_bias))
        return model

    def _load_weights(self, weights_file):
        self.load_file = weights_file
        if os.path.exists(weights_file):
            print('Loading pre-existed weights file')
            self.siamese_net.load_weights(weights_file)
            return True
        return False

    def fit(self, weights_file, train_path, validation_size, batch_size, epochs):
        with open(train_path, 'rb') as f:
            x_train, y_train, names = pickle.load(f)

        x_train_0, x_val_0, y_train_0, y_val_0 = train_test_split(x_train[0], y_train,
                                                                    test_size=validation_size,
                                                                    random_state=self.seed)
        x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(x_train[1], y_train,
                                                                    test_size=validation_size,
                                                                    random_state=self.seed)
        x_train_0 = np.array(x_train_0, dtype='float64')
        x_val_0   = np.array(x_val_0,   dtype='float64')
        x_train_1 = np.array(x_train_1, dtype='float64')
        x_val_1   = np.array(x_val_1,   dtype='float64')
        x_train = [x_train_0, x_train_1]
        x_val   = [x_val_0,   x_val_1]

        if not (np.array_equal(y_train_0, y_train_1) and np.array_equal(y_val_0, y_val_1)):
            raise Exception("y train lists or y validation list do not equal")

        y_train_both = np.array(y_train_0, dtype='float64')
        y_val_both   = np.array(y_val_0,   dtype='float64')

        weights_dir = os.path.dirname(weights_file)
        if weights_dir and not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        if not self._load_weights(weights_file=weights_file):
            print('No such pre-existed weights file')
            print('Beginning to fit the model')
            history = self.siamese_net.fit(x_train, y_train_both, batch_size=batch_size, epochs=epochs,
                                           validation_data=(x_val, y_val_both), verbose=1)
            self.siamese_net.save_weights(self.load_file)
            self._plot_history(history)  # <-- plot dipanggil di sini

        loss, accuracy = self.siamese_net.evaluate(x_val, y_val_both, batch_size=batch_size)
        print(f'Loss on Validation set: {loss}')
        print(f'Accuracy on Validation set: {accuracy}')

# Plot Loss dan Akurasi
    def _plot_history(self, history):
        epochs = range(1, len(history.history['loss']) + 1)

        plt.figure(figsize=(12, 5))

        # --- Grafik Loss ---
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history.history['loss'],     'b-o', label='Training Loss')
        plt.plot(epochs, history.history['val_loss'], 'r-o', label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # --- Grafik Accuracy ---
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history.history['accuracy'],     'b-o', label='Training Accuracy')
        plt.plot(epochs, history.history['val_accuracy'], 'r-o', label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'training_plot.png'), dpi=150)
        plt.show()
        print(f"Plot disimpan di: {os.path.join(output_path, 'training_plot.png')}")
        
    def evaluate(self, test_file, batch_size, analyze=False):
        with open(test_file, 'rb') as f:
            x_test, y_test, names = pickle.load(f)

        print(f'Available Metrics: {self.siamese_net.metrics_names}')

        y_test    = np.array(y_test, dtype='float64')
        x_test[0] = np.array(x_test[0], dtype='float64')
        x_test[1] = np.array(x_test[1], dtype='float64')

        # Evaluasi model
        loss, accuracy = self.siamese_net.evaluate(
            x_test,
            y_test,
            batch_size=batch_size
            )

        # Prediksi untuk Confusion Matrix
        y_pred_prob = self.siamese_net.predict(x_test)

        # Konversi probabilitas ke kelas 0 atau 1
        y_pred = (y_pred_prob > 0.5).astype("int32")

        # Buat confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print("\nConfusion Matrix:")
        print(cm)

        # Visualisasi confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Different Person", "Same Person"]
            )

        plt.figure(figsize=(6, 6))
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")

        # Simpan confusion matrix
        cm_path = os.path.join(output_path, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150)

        plt.show()

        print(f"Confusion matrix disimpan di: {cm_path}")

        if analyze:
            self._analyze(x_test, y_test, names)

        return loss, accuracy

    def _analyze(self, x_test, y_test, names):
        best_class_0_prob  = 1;  best_class_0_name  = None
        worst_class_0_prob = 0;  worst_class_0_name = None
        best_class_1_prob  = 0;  best_class_1_name  = None
        worst_class_1_prob = 1;  worst_class_1_name = None
        prob = self.siamese_net.predict(x_test)
        for pair_index in range(len(names)):
            name      = names[pair_index]
            y_pair    = y_test[pair_index]
            pair_prob = prob[pair_index][0]
            if y_pair == 0:
                if pair_prob < best_class_0_prob:
                    best_class_0_prob = pair_prob;  best_class_0_name = name
                if pair_prob > worst_class_0_prob:
                    worst_class_0_prob = pair_prob; worst_class_0_name = name
            else:
                if pair_prob > best_class_1_prob:
                    best_class_1_prob = pair_prob;  best_class_1_name = name
                if pair_prob < worst_class_1_prob:
                    worst_class_1_prob = pair_prob; worst_class_1_name = name

        print(f'correct classification for different people, y=0, prediction->0, name: {best_class_0_name} | prob: {best_class_0_prob}')
        print(f'misclassification for different people, y=0, prediction->1, name: {worst_class_0_name} | prob: {worst_class_0_prob}')
        print(f'correct classification for same people, y=1, prediction->1, name: {best_class_1_name} | prob: {best_class_1_prob}')
        print(f'misclassification for same people, y=1, prediction->0, name: {worst_class_1_name} | prob: {worst_class_1_prob}')


print("Loaded Siamese Network")


# Main Code
data_path   = r'/Users/dhiska/Documents/KULIAH/Bimbingan/Data dan Kode/Dataset baru/Dataset baru'
output_path = r'/Users/dhiska/Documents/KULIAH/Bimbingan/Data dan Kode/Dataset baru/Dataset baru'

train_name = 'pairsDevTrain'
test_name  = 'pairsDevTest'

WIDTH = HEIGHT = 105
CELLS = 1


def initialize_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run():
    train_path   = os.path.join(output_path, 'train.pickle')
    test_path    = os.path.join(output_path, 'test.pickle')
    weights_path = os.path.join(output_path, 'model.weights.h5')

    print("Loading dataset...")

    loader = DataLoader(WIDTH, HEIGHT, CELLS, data_path, train_path)
    loader.load(train_name)

    loader = DataLoader(WIDTH, HEIGHT, CELLS, data_path, test_path)
    loader.load(test_name)

    print("Building model...")

    model = SiameseNetwork(
        seed=0,
        width=WIDTH,
        height=HEIGHT,
        cells=CELLS,
        loss="binary_crossentropy",
        metrics=['accuracy'],
        optimizer=Adam(learning_rate=0.00005),
        dropout_rate=0.4
    )

    print("Training...")

    model.fit(
        weights_file=weights_path,
        train_path=train_path,
        validation_size=0.2,
        batch_size=32,
        epochs=50
    )

    print("Evaluating...")

    loss, acc = model.evaluate(
        test_file=test_path,
        batch_size=32,
        analyze=True
    )

    print("Final Result:")
    print("Loss:", loss)
    print("Accuracy:", acc)


if __name__ == "__main__":
    start = time.time()
    run()
    print("Time:", time.time() - start)