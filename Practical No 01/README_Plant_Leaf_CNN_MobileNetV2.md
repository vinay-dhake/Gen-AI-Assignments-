# 🌱 Plant Leaf Disease Classification using Custom CNN and MobileNetV2

A practical image-classification project comparing a **Custom Convolutional Neural Network (CNN) trained from scratch** with **MobileNetV2 transfer learning and fine-tuning** for plant-leaf disease classification.

> **Practical Assignment 1 — Image Classification using CNN and Transfer Learning**

---

## 👤 Student Details

- **Student Name:** Vinay Sandip Dhake
- **Student ID:** 202401110031
- **Assignment:** Practical Assignment 1
- **Framework:** TensorFlow / Keras
- **Environment:** Google Colab
- **GPU used:** NVIDIA T4

---

## 🎯 Project Objective

The objective of this practical is to implement an end-to-end image-classification pipeline and compare two approaches:

1. **Custom CNN trained from scratch**
2. **MobileNetV2 pretrained on ImageNet**, followed by transfer learning and fine-tuning

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training time
- Number of parameters
- Accuracy and loss curves
- Prediction examples

The main research question is:

> **Does MobileNetV2 transfer learning and fine-tuning provide better plant-leaf disease classification than a CNN trained from scratch?**

---

## 🌿 Dataset

### PlantVillage Dataset

This project uses the **PlantVillage** plant-leaf image dataset.

The full PlantVillage dataset contains **54,303 images across 38 categories** in the TensorFlow Datasets catalog.

For this classroom experiment, four classes were selected to keep the experiment balanced and computationally manageable.

### Selected Classes

| Class | Description |
|---|---|
| `Tomato___Early_blight` | Tomato leaf affected by Early Blight |
| `Tomato___Late_blight` | Tomato leaf affected by Late Blight |
| `Tomato___healthy` | Healthy tomato leaf |
| `Potato___Early_blight` | Potato leaf affected by Early Blight |

A maximum of **500 images per class** was selected, giving:

**2,000 images total**

The classes were selected in equal numbers so that the experiment was balanced.

---

## 🔬 What is Being Classified?

The model receives a leaf image and predicts **one of four classes**.

For example:

```text
Input:
Tomato leaf image
        ↓
CNN / MobileNetV2
        ↓
Prediction
        ↓
Tomato Early Blight
```

The model learns visual patterns such as:

- edges
- textures
- colors
- shapes
- lesion patterns
- spatial arrangements

It then uses these learned features to classify unseen leaf images.

---

## 🧠 Early Blight vs Late Blight

### Early Blight

Early Blight commonly produces brown lesions and may show characteristic **target-like concentric rings**.

### Late Blight

Late Blight commonly produces **larger, irregular, dark or water-soaked lesions** and can spread rapidly under favorable conditions.

The classifier does not use a single visual feature. CNNs learn combinations of visual patterns from the training images.

---

# 🏗️ Project Pipeline

```text
PlantVillage Dataset
        │
        ▼
Select 4 Classes
        │
        ▼
Data Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
70% Train / 15% Validation / 15% Test
        │
        ├─────────────────────────────┐
        ▼                             ▼
 Custom CNN                    MobileNetV2
 trained from scratch          ImageNet pretrained
        │                             │
        ▼                             ▼
   CNN Training                Feature Extraction
                                      │
                                      ▼
                               Fine-Tuning
        │                             │
        └──────────────┬──────────────┘
                       ▼
                 Evaluation
                       │
                       ▼
 Accuracy / Precision / Recall / F1
 Confusion Matrix / Training Time
 Parameter Count / Predictions
                       │
                       ▼
              Comparative Conclusion
```

---

# 🧪 Data Preprocessing

The notebook performs the following preprocessing:

- Images are resized to **224 × 224**
- Images are converted to 3-channel RGB format
- Data is loaded using a `tf.data` pipeline
- Training data is shuffled
- Batching is performed with batch size **32**
- Prefetching is used for efficient input processing

### Data Augmentation

The training pipeline applies:

- Random horizontal flip
- Random rotation
- Random zoom
- Random translation

The purpose of augmentation is to introduce realistic image variation and reduce overfitting.

---

# 🧩 Train / Validation / Test Split

The selected dataset is split using stratified sampling:

| Split | Images |
|---|---:|
| Training | 1,400 |
| Validation | 300 |
| Testing | 300 |
| **Total** | **2,000** |

The same test set is used for both models to make the comparison fair.

---

# 🧱 Model 1 — Custom CNN

The first model is trained completely **from scratch**.

### Architecture

```text
Input: 224 × 224 × 3
        ↓
Data Augmentation
        ↓
Rescaling
        ↓
Conv2D – 32 filters
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D – 64 filters
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D – 128 filters
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Global Average Pooling
        ↓
Dense – 128
        ↓
Dropout – 0.40
        ↓
Softmax – 4 Classes
```

### Training

The Custom CNN uses:

- Adam optimizer
- Sparse categorical cross-entropy
- Early stopping
- ReduceLROnPlateau
- Maximum 15 epochs

The recorded training time was approximately **49.42 seconds**.

---

# 🚀 Model 2 — MobileNetV2 Transfer Learning

MobileNetV2 is a pretrained convolutional neural network originally trained on ImageNet.

It was selected because it is relatively lightweight while providing useful pretrained visual features.

## Stage 1 — Feature Extraction

The MobileNetV2 base was frozen:

```python
base_model.trainable = False
```

Only the new classification head was trained for the four plant-disease classes.

The model had:

- **2,263,108 total parameters**
- **5,124 trainable parameters before fine-tuning**

## Stage 2 — Fine-Tuning

The final portion of the MobileNetV2 base was unfrozen.

The experiment:

- kept earlier layers frozen
- kept Batch Normalization layers frozen
- fine-tuned the final 30 layers
- used a small learning rate of `1e-5`

This allows pretrained features to adapt to the plant-leaf classification task without aggressively destroying the pretrained representations.

---

# 📊 Evaluation Metrics

The two models are evaluated using:

### Accuracy

The proportion of test images classified correctly.

### Precision

Measures how many predicted positive instances are actually correct.

### Recall

Measures how many actual instances are correctly identified.

### F1-score

The harmonic mean of precision and recall.

### Confusion Matrix

Shows the number of correct and incorrect predictions for every class.

---

# 📈 Experimental Results

The executed notebook produced the following test-set results:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **Custom CNN** | **0.2500** | **0.0625** | **0.2500** | **0.1000** |
| **MobileNetV2** | **0.9400** | **0.9406** | **0.9400** | **0.9398** |

### Training and Complexity

| Model | Training Time | Total Parameters | Final Trainable Parameters |
|---|---:|---:|---:|
| **Custom CNN** | 49.423 s | 111,172 | 111,172 |
| **MobileNetV2** | 74.621 s | 2,263,108 | 1,515,844 |

> These values are the actual recorded outputs in the submitted notebook. They should not be replaced with estimated or fabricated values.

---

# 🏆 Result Interpretation

In this experiment, **MobileNetV2 clearly outperformed the Custom CNN** on the test set.

MobileNetV2 achieved:

- **94.00% accuracy**
- **94.06% precision**
- **94.00% recall**
- **93.98% F1-score**

The Custom CNN achieved only:

- **25.00% accuracy**
- **6.25% precision**
- **25.00% recall**
- **10.00% F1-score**

The results suggest that pretrained ImageNet features provided a much stronger starting representation for this task than learning all visual features from scratch with the small custom CNN.

However, MobileNetV2 used substantially more parameters than the custom CNN. Therefore, the comparison demonstrates an important trade-off between **performance and model complexity**.

---

# 🔍 Feature Map Visualization

The notebook visualizes intermediate feature maps from three convolutional layers of the Custom CNN.

The recorded feature-map dimensions were:

```text
Conv Layer 1 → (1, 224, 224, 32)
Conv Layer 2 → (1, 112, 112, 64)
Conv Layer 3 → (1, 56, 56, 128)
```

Feature maps help visualize the intermediate representations learned by convolutional layers.

Generally:

- Early layers learn low-level features such as edges and textures.
- Deeper layers learn more complex visual patterns.

---

# 🖼️ Prediction Demonstration

The notebook selects eight random test images and displays:

- Actual class
- Predicted class
- Prediction confidence

This provides a visual demonstration of how MobileNetV2 performs on unseen test images.

---

# 📚 Research Paper

### Selected Paper

**Identification of Plant-Leaf Diseases Using CNN and Transfer-Learning Approach**

Published in:

**Electronics, 2021, 10(12), 1388**

Paper:

https://www.mdpi.com/2079-9292/10/12/1388

### Connection with This Practical

The research paper investigates plant-leaf disease identification using CNN and transfer-learning approaches.

This practical follows the same broad methodology:

- Plant-leaf disease classification
- PlantVillage dataset
- CNN-based image classification
- Transfer learning
- Pretrained CNN models
- Evaluation using classification metrics
- Comparative analysis

The practical is a **smaller classroom implementation**, rather than an exact reproduction of every experiment in the paper.

The notebook deliberately uses four selected classes instead of reproducing the complete research setup.

---

# 🆚 Research Paper vs This Practical

| Aspect | Research Paper | This Practical |
|---|---|---|
| Application | Plant disease classification | Plant disease classification |
| Dataset | PlantVillage | PlantVillage |
| CNN approach | Yes | Custom CNN |
| Transfer learning | Yes | MobileNetV2 |
| Fine-tuning | Transfer-learning methodology | Yes |
| Classes | Research setup | 4 selected classes |
| Evaluation | Classification metrics | Accuracy, Precision, Recall, F1 |
| Confusion Matrix | Yes | Yes |
| Model Comparison | Multiple approaches | Custom CNN vs MobileNetV2 |

---

# 💡 What Was Learned

This project demonstrates that:

1. CNNs can automatically learn useful visual features from leaf images.
2. Data augmentation can improve the diversity of training examples.
3. Transfer learning allows a model to reuse knowledge learned from a large dataset.
4. Fine-tuning allows pretrained features to adapt to the target dataset.
5. Accuracy alone is not sufficient for evaluating a classification model.
6. Precision, recall, F1-score and confusion matrices provide additional insight.
7. A pretrained model can significantly outperform a small CNN trained from scratch on this experiment.
8. Model selection should consider both predictive performance and computational complexity.
9. Results on a controlled dataset such as PlantVillage may not directly represent performance on real-world field photographs.

---

# ⚠️ Limitations

### Dataset Limitation

Only four PlantVillage classes were used instead of all 38 categories.

### Controlled Images

PlantVillage images are relatively controlled compared with real agricultural field photographs. Real-world images may contain:

- complex backgrounds
- different lighting
- multiple leaves
- different camera quality
- varying disease severity

### Model Limitation

The Custom CNN performed poorly in this recorded experiment. This highlights the benefit of pretrained representations for the selected task, but it also means the custom architecture could be improved through:

- better architecture design
- more training
- hyperparameter tuning
- improved regularization
- better class/data sampling

---

# 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- NVIDIA T4 GPU

---

# ▶️ How to Run

## Option 1 — Google Colab

1. Open the `.ipynb` notebook in Google Colab.
2. Select:

```text
Runtime → Change runtime type → T4 GPU
```

3. Run the notebook cells from top to bottom.
4. The notebook downloads the PlantVillage data directly.
5. The dataset is extracted automatically.
6. Four classes are selected.
7. Both models are trained.
8. Evaluation tables and graphs are generated.

### Important

This notebook intentionally **does not depend on `tensorflow_datasets` / `tfds.load()`**. The PlantVillage data is downloaded directly and read from folders.

---

# 📁 Repository Structure

A recommended GitHub structure is:

```text
Practical No 01/
│
├── README.md
│
└── 202401110031_Practical_Assignment_1_Plant_Leaf_CNN_MobileNetV2.ipynb
```

If you later add screenshots or result images:

```text
Practical No 01/
│
├── README.md
├── 202401110031_Practical_Assignment_1_Plant_Leaf_CNN_MobileNetV2.ipynb
│
└── results/
    ├── cnn_accuracy_loss.png
    ├── mobilenet_accuracy_loss.png
    ├── cnn_confusion_matrix.png
    ├── mobilenet_confusion_matrix.png
    └── model_comparison.png
```

---

# 🎓 Viva — Quick Answers

### What is image classification?

Image classification is the process of assigning an image to one predefined class. Here, the model assigns a leaf image to one of four plant/disease classes.

### Why CNN?

CNNs are effective for images because convolutional filters learn spatial features such as edges, textures and patterns.

### What is transfer learning?

Transfer learning reuses knowledge learned by a model on a large dataset for a new target task.

### Why MobileNetV2?

MobileNetV2 is a lightweight pretrained CNN that provides useful pretrained visual features and is suitable for transfer-learning experiments.

### What is fine-tuning?

Fine-tuning means unfreezing selected pretrained layers and training them with a small learning rate so that the pretrained features adapt to the target dataset.

### Why did MobileNetV2 perform better?

The pretrained MobileNetV2 already had useful visual representations learned from ImageNet, while the custom CNN had to learn its features from scratch.

### Why did we use four classes?

Four balanced classes make the classroom experiment manageable while still demonstrating multiclass classification and model comparison.

### What is the difference between Early Blight and Late Blight?

Early Blight commonly shows brown lesions that can have target-like concentric rings. Late Blight commonly produces larger, irregular, dark or water-soaked lesions.

### Which model performed better?

**MobileNetV2 performed better in this experiment**, achieving 94% test accuracy and approximately 0.94 F1-score.

---

# 👨‍💻 Author

**Vinay Sandip Dhake**

Student ID: **202401110031**

GitHub Repository:

https://github.com/vinay-dhake/Gen-AI-Assignments-/tree/main/Practical%20No%2001

---

## 📌 Academic Note

This repository is submitted as **Practical Assignment 1** for academic demonstration and comparative study of CNN-based image classification and transfer learning.

The reported metrics above are taken from the executed notebook. The experiment is intended to demonstrate the methodology and comparative analysis rather than reproduce the research paper's exact experimental results.
