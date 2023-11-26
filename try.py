import nnfs
from nnfs.datasets import vertical_data
from sklearn.model_selection import train_test_split

from NNs.Neural_Networks import Neural_Networks
from Neural_Networks import Neural_Networks
from NNs.Layers import Layer_Dense, Layer_Dropout, Layer_Input
from NNs.Activations import Activation_ReLU, Activation_Sigmoid, Activation_Softmax
from NNs.Losses import Loss_CategoricalCrossentropy


nnfs.init()
n_samples = 100
n_classes = 3

X, y = vertical_data(n_samples, n_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# Our implementation

# Instantiate the model

model = Neural_Networks()

# Add layers
model.add_layer(Layer_Dense(X_train.shape[1], 64))
model.add_layer(Activation_ReLU())
model.add_layer(Layer_Dense(64, 64))
model.add_layer(Activation_ReLU())
model.add_layer(Layer_Dense(64, n_classes))
model.add_layer(Activation_Softmax())


# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    # optimizer=Optimizer_Adam(decay=1e-3),
    optimizer=Optimizer_SGD(learning_rate=0.1),
    accuracy=Accuracy_Categorical()
)


# Finalize the model
model.finalize()

# Train the model
model.train(X_train, y_train, validation_data=(X_test, y_test),
            epochs=100, batch_size=128, print_every=128*5)


# model.evaluate(X_test, y_test)
predictions = np.argmax(model.predict(X_test), axis=1)
print(predictions, y_test)
