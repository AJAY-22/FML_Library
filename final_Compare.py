# For Dataset
import nnfs
from nnfs.datasets import vertical_data, spiral_data
from sklearn.model_selection import train_test_split


from NNs.Model import Model
from NNs.Accuracies import Accuracy, Accuracy_Categorical
from NNs.Activations import Activation_ReLU, Activation_Softmax
from NNs.Layers import Layer_Dense
from NNs.Losses import Loss, Loss_CategoricalCrossentropy
from NNs.Optimizers import Optimizer_SGD



nnfs.init()

n_samples = 1000
n_classes = 3

# set datset to 0 (for vertical_data) and 
# 1 (for spiral_data)
dataset = 0
if dataset:
    X, y = spiral_data(n_samples, n_classes)
else:
    X, y = vertical_data(n_samples, n_classes)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.33)

model = Model()
model.add(Layer_Dense(X.shape[1], 3))
model.add(Activation_ReLU())
model.add(Layer_Dense(6, 6))
model.add(Activation_ReLU())
model.add(Layer_Dense(6, n_classes))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_SGD(learning_rate=0.01),
    accuracy=Accuracy_Categorical()
)

model.train(X, y, validation_data=(X_test, y_test),
            epochs=320, batch_size=30, print_every=100)

model.evaluate()