from sklearn.model_selection import train_test_split

import numpy as np 
import nnfs
from nnfs.datasets import spiral_data, vertical_data  
from Layers import Layer_Dense
from Activations import Activation_ReLU, Activation_Softmax
from Losses import Loss_CategoricalCrossentropy
from Optimizers import Optimizer_SGD, Optimizer_Adam
from Accuracy import Accuracy_Categorical
from Model import Model

nnfs.init()
n_samples = 10000
n_classes = 3
X, y = vertical_data(n_samples, n_classes)  
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.33)

# Our implementation
# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(X_train.shape[1], 3))
model.add(Activation_ReLU())
# model.add(Layer_Dense(64, 64))
# model.add(Activation_ReLU())
model.add(Layer_Dense(3, n_classes))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
loss=Loss_CategoricalCrossentropy(),
# optimizer=Optimizer_Adam(decay=1e-3),
optimizer=Optimizer_SGD(),
accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()
# Train the model
model.train(X_train, y_train, validation_data=(X_test, y_test),
epochs=100, batch_size=128, print_every=100)

model.evaluate(X_test, y_test)


print("=================================================================")
# TF Implementation
import tensorflow as tf
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(3, activation='relu'),
#   tf.keras.layers.Dense(64, activation='relu'),
#   tf.keras.layers.Dense(3, activation='relu'),
  tf.keras.layers.Dense(n_classes)
])

predictions = model(X_train).numpy()
tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train, predictions).numpy()

op = tf.keras.optimizers.Adam(
    weight_decay = 5e-5
)
model.compile(optimizer=op,
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)
model.evaluate(X_test,  y_test, verbose=2)
