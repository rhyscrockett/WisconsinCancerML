import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define Neurons
input_neurons = 30
hidden_neurons = 15
output_neurons = 2

# Define Network Parameters
learning_rate = 0.01
epochs = 2000
error = []

# Define Weights
w1 = 2 * np.random.random((input_neurons, hidden_neurons)) - 1
w2 = 2 * np.random.random((hidden_neurons, output_neurons)) - 1

# Define Bias'
b1 = np.full((1, hidden_neurons), 1)
b2 = np.full((1, output_neurons), 1)

# Preparing Dataset
wdbc = pd.read_csv('wdbc.data', header=None)

wdbc.columns = ["ID Number", "Diagnosis",
                "Radius (M)", "Texture (M)", "Perimeter (M)", "Area (M)", "Smoothness (M)", "Compactness (M)", "Concavity (M)", "Concave Point (M)", "Symmetry (M)", "Fractal Dimension (M)",
                "Radius (SE)", "Texture (SE)", "Perimeter (SE)", "Area(SE)", "Smoothness (SE)", "Compactness (SE)", "Concavity (SE)", "Concave Points (SE)", "Symmetry (SE)", "Fractal Dimension (SE)",
                "Radius (W)", "Texture (W)", "Perimeter (W)", "Area (W)", "Smoothness (W)", "Compactness (W)", "Concavity (W)", "Concave Points (W)", "Symmetry (W)", "Fractal Dimension (W)"]

# Drop the first column (ID Number)
wdbc = wdbc.drop('ID Number', axis=1)

# Replace the Diagnosis with Binary instead of Character denoting results
wdbc = wdbc.replace({'M': 1, 'B': 0})

# Split data into Input and Output 
X = wdbc.iloc[:, 1:]
Y = wdbc["Diagnosis"]

# Normalisation (Min-Max) using the input features
X_min = X.min() * 0.8
X_max = X.max() * 1.2

wdbc_normalised = (X - X_min) / (X_max - X_min)
wdbc_normalised.insert(0, 'Diagnosis', Y)

# Split the Data-set
train, test = np.split(wdbc_normalised.sample(frac=1), [int(0.70 * len(wdbc_normalised))])
#print("Training Set: ", train.shape)
#print("Testing Set: ", test.shape)

X_train = train.iloc[:, 1:]
Y_train = train["Diagnosis"]
X_test = test.iloc[:, 1:]
Y_test = test["Diagnosis"]

# Define Activation Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)
    
### Initalise Training Network
print("Training Network...")

# Define Input Layer (Training)
input_layer = np.array(X_train)

# Outputs (Training)
outputs = np.array(Y_train).reshape(-1, 1)

# Forward Propagation
for epoch in range(epochs):

    # Define Hidden Layer
    sum_synapse_hidden = np.dot(input_layer, w1) + b1
    hidden_layer = sigmoid(sum_synapse_hidden)

    # Define Output Layer
    sum_synapse_output = np.dot(hidden_layer, w2) + b2
    output_layer = sigmoid(sum_synapse_output)

    # Calculate Loss (error)
    error_output_layer = outputs - output_layer
    average_error = np.mean(abs(error_output_layer))
    if epoch % 100 == 0:
        print("[Error Avg: " + str(average_error) + "]" + " - Epoch " + str(epoch))
        error.append(average_error)

    # Back Propagation
    # Delta Output Layer ()
    delta_output_layer = error_output_layer * sigmoid_derivative(output_layer)
    delta_output_weight = delta_output_layer.dot(w2.T)

    # Delta Hidden Layer ()
    delta_hidden_layer = delta_output_weight * sigmoid_derivative(hidden_layer)

    # Update Weights
    w1 = w1 + (input_layer.T.dot(delta_hidden_layer) * learning_rate)
    w2 = w2 + (hidden_layer.T.dot(delta_output_layer) * learning_rate)

### Initalise Testing Network
# Inputs
input_layer = np.array(X_test)

# Outputs
outputs = np.array(Y_test)

# Define Hidden Layer
sum_synapse_hidden = np.dot(input_layer, w1) + b1
hidden_layer = sigmoid(sum_synapse_hidden)

# Define Output Layer
sum_synapse_output = np.dot(hidden_layer, w2) + b2
output_layer = sigmoid(sum_synapse_output)

### Up to this point the weights generated from testing are used to cycle through once with the testing dataset
### functions would solve the need for reused code

# Testing
print("\nTesting Network...")
prediction = output_layer # y_pred
prediction = [1 if i[0] >= 0.5 else 0 for i in prediction] # predict using testing set
prediction = np.array(prediction)
score = np.sum(prediction == outputs) # accuracy score
print("Acc: " + str(round(score/len(outputs), 3) * 100) + "%")
#print("Acc: ", cnt/len(outputs) * 100, "%")

cm = confusion_matrix(outputs, prediction)
print(cm)
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')

ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
plt.show()
