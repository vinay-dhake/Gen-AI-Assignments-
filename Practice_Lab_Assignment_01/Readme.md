# Neural Network Implementation from Scratch

## Practice Lab – Generative AI Lab

**Student Name:** Vinay Sandip Dhake
**PRN:** 202401110031
**Class:** T.Y. BTech
**Batch:** A2
**Course:** Generative AI Lab
**Department:** CSE – Artificial Intelligence and Machine Learning
**Date of Submission:** 15 Aug 2026

---

## 📌 Project Overview

This project implements a simple **feedforward neural network from scratch using Python and NumPy**.

The main purpose of this practice lab is to understand the fundamental working of a neural network by implementing its major components manually rather than using deep learning frameworks such as TensorFlow, PyTorch, or Keras.

The implementation covers:

* Forward propagation
* Sigmoid activation
* Softmax activation
* Categorical Cross-Entropy Loss
* Backpropagation
* Gradient computation
* Gradient Descent optimization
* Model training
* Accuracy evaluation
* Confusion matrix analysis
* Comparison with traditional machine learning models

---

## 🎯 Objectives

The main objectives of this practice lab are:

1. To understand the basic architecture of a feedforward neural network.
2. To implement neural-network operations using NumPy.
3. To understand how forward propagation generates predictions.
4. To implement activation functions and their derivatives.
5. To understand and implement categorical cross-entropy loss.
6. To implement backpropagation manually.
7. To update model parameters using gradient descent.
8. To train the neural network on the Iris dataset.
9. To evaluate the trained model using accuracy and a confusion matrix.
10. To compare the neural network with alternative machine-learning models.

---

## 🌸 Dataset

The **Iris dataset** is used for this practice lab.

The dataset contains:

* **150 samples**
* **4 numerical input features**
* **3 target classes**

### Input Features

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

### Target Classes

1. Setosa
2. Versicolor
3. Virginica

The Iris dataset is loaded using `sklearn.datasets.load_iris()`.

---

## 🧩 Problem Definition

This project solves a **multi-class classification problem**.

Given the four measurements of an iris flower, the neural network predicts which of the three species the flower belongs to.

Mathematically:

```text
Input Features
      ↓
Neural Network
      ↓
Class Probabilities
      ↓
Predicted Iris Species
```

---

## 🏗️ Neural Network Architecture

The implemented neural network contains one hidden layer.

```text
Input Layer
4 Neurons
     ↓
Hidden Layer
8 Neurons
Sigmoid Activation
     ↓
Output Layer
3 Neurons
Softmax Activation
     ↓
Predicted Class
```

### Architecture Summary

| Layer        | Neurons | Activation |
| ------------ | ------: | ---------- |
| Input Layer  |       4 | None       |
| Hidden Layer |       8 | Sigmoid    |
| Output Layer |       3 | Softmax    |

The four input neurons correspond to the four Iris features, while the three output neurons correspond to the three target classes.

---

## ⚙️ Methodology

### 1. Data Preprocessing

The dataset is divided into training and testing sets using an 80:20 split.

```text
Training Samples: 120
Testing Samples: 30
```

The input features are standardized using `StandardScaler`.

The target labels are converted into one-hot encoded vectors because the problem contains three output classes.

---

### 2. Forward Propagation

During forward propagation, the input data passes through the network.

The hidden-layer weighted sum is calculated as:

```text
Z1 = X · W1 + b1
```

The Sigmoid activation is then applied:

```text
A1 = sigmoid(Z1)
```

The output-layer weighted sum is calculated as:

```text
Z2 = A1 · W2 + b2
```

Finally, Softmax converts the output values into class probabilities:

```text
A2 = softmax(Z2)
```

---

### 3. Activation Functions

#### Sigmoid

The hidden layer uses the Sigmoid activation function:

```text
σ(x) = 1 / (1 + e^(-x))
```

Its derivative is:

```text
σ'(x) = σ(x)(1 - σ(x))
```

#### Softmax

The output layer uses Softmax because this is a multi-class classification problem.

Softmax converts the output values into probabilities whose sum is equal to 1.

---

### 4. Loss Function

**Categorical Cross-Entropy Loss** is used to measure the difference between the actual class and the predicted class probabilities.

```text
Loss = -(1/n) Σ y_true × log(y_pred)
```

A small epsilon value is added while calculating the logarithm to avoid numerical problems caused by `log(0)`.

---

### 5. Backpropagation

Backpropagation is implemented manually to calculate the gradients of the loss with respect to the weights and biases.

For the output layer:

```text
dZ2 = A2 - Y
```

The gradients are calculated for:

```text
dW2
db2
dW1
db1
```

The error is propagated from the output layer back toward the hidden layer using the chain rule.

---

### 6. Gradient Descent

Gradient Descent is used to update the model parameters.

The update rule is:

```text
W = W - learning_rate × dW
b = b - learning_rate × db
```

The learning rate used in this implementation is:

```text
Learning Rate = 0.1
```

The model is trained for:

```text
Epochs = 1000
```

---

## 🧪 Implementation Details

The neural network is implemented as a custom Python class:

```text
NeuralNetwork
```

The class contains methods for:

* `__init__()` – initializes weights, biases, and learning rate
* `forward()` – performs forward propagation
* `backward()` – calculates gradients and updates parameters
* `train()` – trains the network for multiple epochs
* `predict()` – generates predicted class labels

All core neural-network computations are implemented using **NumPy**.

---

## 📉 Training Performance

The training loss decreases progressively during training.

The loss starts at approximately:

```text
1.2333
```

and decreases as the number of epochs increases.

Selected training-loss values:

| Epoch |   Loss |
| ----: | -----: |
|     0 | 1.2333 |
|   100 | 0.5908 |
|   200 | 0.4231 |
|   300 | 0.3483 |
|   400 | 0.2967 |
|   500 | 0.2555 |
|   600 | 0.2216 |
|   700 | 0.1943 |
|   800 | 0.1726 |
|   900 | 0.1554 |

The decreasing loss indicates that the neural network is learning from the training data.

---

## 📊 Model Evaluation

The trained neural network is evaluated using training accuracy, testing accuracy, and a manually implemented confusion matrix.

### Accuracy

| Metric            |      Result |
| ----------------- | ----------: |
| Training Accuracy |  **96.67%** |
| Testing Accuracy  | **100.00%** |

The model correctly classified all 30 samples in the held-out test set used in the experiment.

---

## 🔢 Confusion Matrix

The manually implemented confusion matrix is:

```text
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

The rows represent the actual classes and the columns represent the predicted classes.

The diagonal values represent correctly classified samples.

The result shows that all test samples were correctly classified in this particular train/test split.

---

## 🔍 Comparison with Alternative Models

To understand the neural network's performance in context, it was compared with:

* Logistic Regression
* Decision Tree
* Support Vector Machine (SVM)

The obtained comparison was:

| Model                    |    Accuracy |
| ------------------------ | ----------: |
| Neural Network (Scratch) | **100.00%** |
| Logistic Regression      |      96.67% |
| SVM                      |      96.00% |
| Decision Tree            |      94.00% |

The comparison shows that all models perform well on the Iris dataset.

> **Note:** The from-scratch neural network accuracy is calculated on the held-out test set, while the baseline models in the current notebook use 10-fold cross-validation. Therefore, these accuracy values should be treated as a contextual comparison rather than a perfectly identical evaluation protocol.

---

## 💡 Analysis and Findings

The training loss decreases steadily during training, indicating that the network successfully learns from the training data.

The neural network achieves high classification accuracy on the Iris dataset, demonstrating that the manually implemented forward propagation, backpropagation, and gradient-descent optimization are working correctly.

The confusion matrix shows correct classification for all samples in the test set for this particular split.

The Iris dataset is relatively small and comparatively simple, so traditional machine-learning algorithms such as Logistic Regression, Decision Tree, and SVM can also achieve high accuracy.

The experiment demonstrates that a single hidden layer containing eight neurons is sufficient to obtain strong results on this dataset.

However, the 100% testing accuracy should be interpreted carefully because the test set contains only 30 samples. A different train/test split could produce a different result.

For larger and more complex datasets, neural networks can be extended using:

* Additional hidden layers
* More neurons
* ReLU activation
* Mini-batch or stochastic gradient descent
* Advanced optimizers such as Adam
* Regularization techniques
* Larger and more diverse datasets

---

## 📈 Visualizations

The notebook contains the following visualizations:

### 1. Training Loss Curve

Shows the change in Cross-Entropy Loss across training epochs.

### 2. Confusion Matrix

Shows the classification performance for each Iris class.

### 3. Model Accuracy Comparison

Compares the neural network with Logistic Regression, Decision Tree, and SVM.

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **Jupyter Notebook / Google Colab**

### Important

The neural network itself is implemented from scratch using NumPy.

No deep-learning frameworks such as:

* TensorFlow
* PyTorch
* Keras

are used for the neural-network implementation.

---


---

## ▶️ How to Run the Project

### Option 1: Google Colab

1. Open the `.ipynb` file in Google Colab.
2. Run the notebook cells sequentially.
3. Verify that all cells execute without errors.
4. Review the generated graphs and evaluation results.

### Option 2: Jupyter Notebook

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Then launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Vinay_Dhake_GenerativeAILabAssignment.ipynb
```

Run all cells sequentially.

---

## 📌 Key Learning Outcomes

Through this practice lab, the following concepts were implemented and understood:

* Neural network architecture
* Weight and bias initialization
* Forward propagation
* Activation functions
* Sigmoid derivative
* Softmax
* Cross-Entropy Loss
* Backpropagation
* Gradient calculation
* Gradient Descent
* Model training
* Classification accuracy
* Confusion matrix
* Model comparison
* Interpretation of model performance

---

## 🚀 Future Improvements

The implementation can be extended in the future by adding:

1. ReLU activation function.
2. Multiple hidden layers.
3. Mini-batch gradient descent.
4. Stochastic Gradient Descent.
5. Adam optimizer.
6. Learning-rate scheduling.
7. Regularization techniques such as L1 and L2.
8. Validation-set monitoring.
9. Hyperparameter tuning.
10. Application to larger datasets.

---

## 📚 Assignment Requirements Covered

This project addresses the major requirements of the practice assignment:

* Dataset description
* Classification task definition
* Neural network architecture
* Activation functions
* Forward propagation
* Backpropagation
* Loss function
* Gradient descent
* Model training
* Model evaluation
* Visualizations
* Model-performance metrics
* Alternative model comparison
* GitHub repository
* Documentation

The assignment also requires the notebook, dataset/link, visualizations where applicable, performance metrics/screenshots, and a README as submission materials.

---

## 👨‍💻 Author

**Vinay Sandip Dhake**

B.Tech – Computer Science and Engineering
Artificial Intelligence and Machine Learning
MIT Academy of Engineering, Alandi, Pune

**PRN:** 202401110031
**Class:** T.Y. BTech
**Batch:** A2

---

## 🔗 GitHub Repository

**GitHub Repository:**
https://github.com/vinay-dhake/Gen-AI-Assignments-/tree/main/Practice_Lab_Assignment_01


---

## 📜 Academic Integrity

I, **Vinay Sandip Dhake**, confirm that the work submitted in this assignment is my own and has been completed following academic integrity guidelines.

The implementation is intended for academic learning and demonstrates the fundamental concepts of neural-network implementation from scratch.
