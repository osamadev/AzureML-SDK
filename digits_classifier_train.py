import json
import os
from azureml.core import Run
from azureml.core.model import Model
import pickle
import keras
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense,Dropout, Flatten, MaxPooling2D, Conv2D, BatchNormalization

def preprocess_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    
    #normalize pixel values 
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX = trainX / 255.0
    testX = testX / 255.0
    
    # reshape dataset to have a single channel (grey scale images)
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    # one hot encode target labels
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    
    return trainX, trainY, testX, testY

x_train, y_train, x_test, y_test = preprocess_dataset()

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout', type=float)
parser.add_argument('--hidden', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)

args = parser.parse_args()

model = keras.models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(args.hidden, activation='relu', kernel_initializer='he_uniform'))

if args.dropout is not None and args.dropout < 1:
    model.add(Dropout(args.dropout))
    
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

# compile model
opt = SGD(lr=args.learning_rate, momentum=args.momentum)

model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

os.makedirs('outputs',exist_ok=True)
model.save('outputs/digit_classifier_model.hdf5')

# Log metrics
run = Run.get_context()
run.log('Test Loss', score[0])
run.log('Accuracy', score[1])
