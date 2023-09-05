import tensorflow as tf

from keras import Sequential
from keras.layers import Dropout, MaxPooling2D, Conv2D, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
#from matplotlib import pyplot
from numpy.random import seed
seed = 1316
import numpy as np
from scikeras.wrappers import KerasClassifier
from keras.datasets import cifar10
from pulearn import ElkanotoPuClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

np.random.seed(seed)

def runCnnPU():
    # Load CIFAR10 dataset
    (x_train, y_train), (x_test, y_test) = load_cifar_dataset_binary(classA=3, classB=5)

    # Normalize the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Cross validate
    beginCrossValidation(x_train, y_train, N=4, k=5, pu=True)
    return

def beginCrossValidation(x_train, y_train, N, k, pu=True):
    # Create a dataframe for our results
    columns = ['conv_ratio']
    for i in range(0, k):
        columns.append('fold' + str(i + 1))
    columns.append('mean')
    columns.append('std')
    pd.set_option('display.max_columns', None)
    df = pd.DataFrame(columns=columns)

    if (pu):
        # Cross validate N times for each convertion ratio. The ratio is increased by 10% in each step
        for conv_ratio in range(5, 95, 10):
            print(f"Current conversion ratio: {conv_ratio}%")
            for n in range(0, N):
                print(f"Model: {n} out of {N}")
                curr_accuracies, mean_accuracy, standard_dev = cross_validate(x_train, y_train, K=k, conv_ratio=conv_ratio / 100, pu=True)

                curr_accuracies.insert(0, conv_ratio)
                curr_accuracies.append(mean_accuracy)
                curr_accuracies.append(standard_dev)
                # print("Length of curr_accuracies:", len(curr_accuracies))
                # print("Number of columns in df:", len(df.columns))
                # print("Columns in df:", df.columns.tolist())
                if len(curr_accuracies) != len(df.columns):
                    print("Mismatch in length! curr_accuracies:", len(curr_accuracies), "df.columns:", len(df.columns))
                df.loc[len(df)] = curr_accuracies



        print(df)
        dt = datetime.now()
        ts = datetime.timestamp(dt)
        df.to_csv('./results' + str(ts) + '.csv')

        return
    else:
        # Cross validate N times.
        for n in range(0, N):
            print(f"Model: {n} out of {N}")
            curr_accuracies, mean_accuracy, standard_dev = cross_validate(x_train, y_train, K=k, pu=False)

            curr_accuracies.insert(0, 0)
            curr_accuracies.append(mean_accuracy)
            curr_accuracies.append(standard_dev)
            df.loc[len(df)] = curr_accuracies

        print(df)
        dt = datetime.now()
        ts = datetime.timestamp(dt)
        df.to_csv('./results' + str(ts) + '.csv')

        return

# Perform k-fold cross validation
def cross_validate(x_train, y_train, K=5, conv_ratio=0.05, pu=True):
    # Create an instance of StratifiedKFold with the desired number of folds
    skf = StratifiedKFold(n_splits=K, random_state=seed, shuffle=True)

    accuracies = []
    fold = 0
    # Loop through the folds and split the data
    for train, test in skf.split(x_train, y_train):
        print(f"Fold {fold}:")

        if (pu):
            # Generate pu training data using the fold's train labels
            y_train_pu = convert_to_PU(y_train[train], conv_ratio=conv_ratio)

            sc_compliant_model = KerasClassifier(model=create_binary_model(), epochs=50, batch_size=64, verbose=1)

            # Create a PU estimator using the wrapped model
            model = ElkanotoPuClassifier(estimator=sc_compliant_model, hold_out_ratio=0.2)

            # Train the model
            model.fit(x_train[train], y_train_pu)

            # Predict using validation data
            val_data, val_labels = x_train[test], y_train[test]
            preds = model.predict(val_data)
            preds = preds.reshape((len(val_labels), 1))
            preds = np.where(preds == 1, 1, 0)  # switch -1s to 0s
            # print(f"  Predict:  preds={preds}")

            # Manually compare the predictions to y_test
            correctPreds = val_data[np.where(preds == val_labels)]
            print("Correct predictions: ", len(correctPreds), ". Accuracy: ", len(correctPreds) / len(val_labels))

        else:
            # Create and fit the model
            model = create_binary_model()
            model.fit(x_train[train], y_train[train], epochs=50, batch_size=64, verbose=1)

            val_data, val_labels = x_train[test], y_train[test]
            preds = model.predict(val_data)
            preds = preds.squeeze()  # Remove singleton dimensions, if any
            preds = np.where(preds >= 0.5, 1, 0)  # Convert to binary labels

            # Manually compare the predictions to y_test
            correctPreds = val_data[preds == val_labels.squeeze()]
            print("Correct predictions: ", len(correctPreds), ". Accuracy: ", len(correctPreds) / len(val_labels))


        accuracies.append(len(correctPreds) / len(val_labels))
        fold += 1

    print('Accuracies table: ', accuracies)
    print("Overall accuracy: ", sum(accuracies) / K)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(accuracies), np.std(accuracies)))
    return accuracies, np.mean(accuracies), np.std(accuracies)


# Loads the CIFAR10 dataset and converts it to binary. Converts all classes.
def load_cifar_full_dataset_binary(positive_class=1):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert labels to binary
    y_train = np.where(y_train == positive_class, 1, 0)
    y_test = np.where(y_test == positive_class, 1, 0)

    return (x_train, y_train), (x_test, y_test)


# Loads the CIFAR10 dataset and converts it to binary. Converts only classes A and B! The rest are discarded.
def load_cifar_dataset_binary(classA=3, classB=5):

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Select only classes A and B
    index_train = np.where((y_train == classA) | (y_train == classB))[0]
    x_train = x_train[index_train]
    y_train = y_train[index_train]

    index_test = np.where((y_test == classA) | (y_test == classB))[0]
    x_test = x_test[index_test]
    y_test = y_test[index_test]

    # Convert labels to binary
    y_train = np.where(y_train == classA, 1, 0)
    y_test = np.where(y_test == classA, 1, 0)

    return (x_train, y_train), (x_test, y_test)


# Convert a binary label array to PU by switching zeroes to -1.
# Also converts a percentage of the positive examples to unlabeled
def convert_to_PU(y_train, conv_ratio):
    # Positive labels
    y_positive = np.where(y_train == 1)

    # Select a ratio of the positive labels to convert to unlabeled
    neg_length = int(conv_ratio * len(y_positive[0]))
    pos_to_neg = np.random.choice(len(y_positive[0]), size=neg_length, replace=False)
    pos_to_neg = y_positive[0][pos_to_neg]

    y_pos = np.random.permutation(pos_to_neg)

    y_train_pu = np.where(y_train == 1, 1, -1)
    y_train_pu[y_pos] = -1

    print(f"Number of Positive class examples: {np.sum(y_train_pu == 1)}")
    print(f"Number of Unlabeled class examples: {np.sum(y_train_pu == -1)}")
    return y_train_pu

def create_binary_model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()
    return model


def normalCNN():
    # binary CNN Architecture: https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa

    # load dataset
    (x_train, y_train), (x_test, y_test) = load_cifar_dataset_binary(classA=3, classB=5) # load_cifar_full_dataset_binary(positive_class=2)

    # Normalize the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Print the number of positive and negative class samples remaining
    print(f"Number of positive class samples: {np.sum(y_train == 1)}")
    print(f"Number of negative class samples: {np.sum(y_train == 0)}")
    print("shapes: ", x_train.shape, y_train.shape)

    # Create and fit the model
    model = create_binary_model()
    # model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test), verbose=1)

    # return
    beginCrossValidation(x_train, y_train, N=4, k=5, pu=False)
    return

    # Predict using validation data
    preds = model.predict(x_test)
    preds = preds.squeeze()  # Remove singleton dimensions, if any
    preds = np.where(preds >= 0.5, 1, 0)  # Convert to binary labels

    # Manually compare the predictions to y_test
    correctPreds = x_test[preds == y_test.squeeze()]
    print("Correct predictions: ", len(correctPreds), ". Accuracy: ", len(correctPreds) / len(y_test))




if __name__ == '__main__':
    runCnnPU()
    # normalCNN()
