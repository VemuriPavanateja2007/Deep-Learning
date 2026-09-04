# Deep Learning — Practicals

A hands-on collection of Deep Learning lab practicals implemented in Python using **TensorFlow**, **Keras**, **NumPy**, and **Matplotlib**. This repository progresses from first-principles neural network building blocks (activation functions, a perceptron implemented from scratch) to full Keras-based deep neural networks trained on real image datasets (MNIST, Fashion-MNIST) — with an emphasis on understanding what's happening *inside* the network, not just calling `model.fit()`.

---

## 📌 About This Repository

Each notebook in `Practicals/` is a self-contained Google Colab practical (each includes an "Open in Colab" badge) that builds up core deep learning concepts step by step:

- How individual **activation functions** behave and why they matter
- How a **single-layer perceptron** works — and precisely why it fails on non-linearly separable problems
- How **forward propagation** moves data through a multi-layer network, layer by layer
- How to build a **feedforward neural network entirely from scratch** using low-level TensorFlow operations (manual weights, `tf.GradientTape`)
- How **regularization techniques** (Dropout, L2 weight decay, He initialization) affect training and generalization in a Keras model

Together, the practicals mirror the typical structure of an academic Deep Learning lab course: theory → from-scratch implementation → framework-based implementation → tuning and regularization.

---

## 🗂️ Repository Structure

```
Deep-Learning/
└── Practicals/
    ├── DL_Pract_1.ipynb   # Visualization of activation functions
    ├── DL_Pract_2.ipynb   # Perceptron from scratch + XOR limitation
    ├── DL_Pract_3.ipynb   # Forward propagation & layer-wise activation analysis
    ├── DL_Pract_4.ipynb   # From-scratch DNN on Fashion-MNIST (low-level TensorFlow)
    ├── DL_Pract_5.ipynb   # Regularized DNN on MNIST (Dropout, L2, He init) using Keras
    └── DL_Pract_6.ipynb   # From-scratch DNN on Fashion-MNIST (low-level TensorFlow)
```

---

## 📖 Practical-by-Practical Breakdown

### 1. Visualization of Activation Functions — `DL_Pract_1.ipynb`
Plots and compares the five core activation functions used in neural networks, over an input range of 400 evenly spaced points from -10 to 10:
- **Linear**
- **ReLU**
- **Sigmoid**
- **Tanh**
- **Softmax**

Built using `tensorflow`, `numpy`, and `matplotlib` — a foundational practical for building intuition on why non-linear activations are necessary in deep networks.

### 2. Perceptron Implementation & the XOR Limitation — `DL_Pract_2.ipynb`
Implements a **single-layer Perceptron from scratch** in NumPy — including weight/bias initialization, the step activation function, and the Perceptron learning rule (weight updates on misclassification).

- Trains and tests the Perceptron on the **AND gate** (a linearly separable problem) — achieving high accuracy
- Trains and tests the same Perceptron on the **XOR gate** (a non-linearly separable problem) — demonstrating its failure
- Includes a written explanation of *why* a single-layer Perceptron cannot solve XOR, motivating the need for multi-layer networks

### 3. Forward Propagation & Layer-wise Activation Analysis — `DL_Pract_3.ipynb`
Builds a small Keras `Sequential` model (Dense(5) → Dense(4) → Dense(2)) and inspects **what each layer actually outputs**:
- Constructs intermediate sub-models to extract the output of every layer for a given input
- Prints the shape and values of each layer's output during forward propagation
- Computes the **average activation value per layer** and visualizes it using a Seaborn bar plot

### 4. From-Scratch Deep Neural Network on Fashion-MNIST — `DL_Pract_4.ipynb`
Implements a feedforward neural network **without using Keras's high-level `Model`/`Sequential` API** — built manually with `tf.Module`:
- Loads and normalizes the **Fashion-MNIST** dataset, reshaping images to flat 784-dim vectors
- Manually initializes weights and biases for a hidden layer (128 units, ReLU) and an output layer (10 classes)
- Defines the loss (`SparseCategoricalCrossentropy`) and optimizer (`SGD`, lr = 0.01) manually
- Trains using a custom loop with `tf.GradientTape` for gradient computation over 10 epochs, batch size 64
- Plots training loss and training accuracy curves over epochs

### 5. Regularized Deep Neural Network on MNIST — `DL_Pract_5.ipynb`
Builds a deeper, regularized network using the **Keras Sequential API** to classify MNIST digits, focusing on generalization techniques:
- Architecture: Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Dense(10, softmax)
- Uses **He Normal initialization** and **L2 kernel regularization (0.001)** on the hidden layers
- Compiled with the **Adam optimizer** (lr = 0.001) and trained for 15 epochs with a 20% validation split
- Evaluates final test accuracy and plots **training vs. validation accuracy** to illustrate the effect of Dropout/L2 on overfitting

### 6. From-Scratch Deep Neural Network on Fashion-MNIST — `DL_Pract_6.ipynb`
> **Note:** This notebook's code is currently identical to Practical 4 (same manual `tf.Module` DNN on Fashion-MNIST). It's included here as-is for completeness — worth revisiting to either differentiate it (e.g. a different architecture, optimizer, or dataset) or consolidate it with Practical 4.

---

## 🧠 Concepts Covered

**Neural Network Fundamentals**
- Activation functions (Linear, ReLU, Sigmoid, Tanh, Softmax) and their behavior
- Perceptron learning rule and the linear separability limitation (XOR problem)
- Forward propagation and layer-wise activation inspection

**Building Networks — From Scratch vs. Framework**
- Manual weight/bias initialization and gradient computation with `tf.GradientTape`
- Custom training loops vs. Keras's built-in `model.fit()`

**Regularization & Generalization**
- Dropout
- L2 weight regularization
- He Normal weight initialization
- Train/validation accuracy comparison to detect overfitting

**Datasets Used**
- MNIST (handwritten digits)
- Fashion-MNIST (clothing image classification)

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab / Jupyter Notebook
- **Core Libraries:**
  - `tensorflow` / `tensorflow.keras` — model building, training, and low-level tensor operations
  - `numpy` — numerical computation, from-scratch perceptron implementation
  - `matplotlib` — training curves and activation function plots
  - `seaborn` — layer-wise activation bar plots (Practical 3)
  - `pandas` — structuring activation data for visualization (Practical 3)

---

## 🚀 Getting Started

### Option 1 — Run in Google Colab (recommended)
Each notebook includes an **"Open in Colab"** badge at the top — click it to run the notebook directly with no local setup. Datasets (MNIST, Fashion-MNIST) are downloaded automatically via `tf.keras.datasets`, so no manual data setup is required.

### Option 2 — Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/VemuriPavanateja2007/Deep-Learning.git
   cd Deep-Learning/Practicals
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib seaborn pandas jupyter
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

4. Run cells sequentially — Practicals 4, 5, and 6 will download MNIST/Fashion-MNIST automatically on first run, so an internet connection is required.

---

## 🎯 Purpose & Learning Outcomes

This repository was built to develop a strong foundational understanding of deep learning by implementing concepts at multiple levels of abstraction rather than only using high-level APIs. Working through these practicals reinforced:

- Why non-linear activation functions are essential for learning complex decision boundaries
- The precise mathematical reason a single-layer Perceptron cannot solve non-linearly separable problems like XOR
- How data actually transforms as it passes through each layer of a neural network
- How to implement gradient-based training manually using `tf.GradientTape`, before relying on Keras's abstractions
- How regularization techniques (Dropout, L2, weight initialization strategy) directly affect the gap between training and validation performance

---

## 🔭 Future Improvements

- [ ] Differentiate or consolidate Practical 4 and Practical 6 (currently identical code)
- [ ] Add a `requirements.txt` for reproducible local setup
- [ ] Add markdown theory cells explaining the math behind each concept before the code
- [ ] Extend Practical 4/6's from-scratch DNN with a validation split and regularization for direct comparison with Practical 5
- [ ] Add a CNN-based practical for image classification to complement the fully-connected architectures here

---

## 👤 Author

**Vemuri Venkata Satya Markandeya Pavanateja**
Final-year B.Sc Artificial Intelligence student, Government College (Autonomous), Rajahmundry

- LinkedIn: [linkedin.com/in/vemurivenkata-satya-markandeyapavanateja-0651a0403](https://www.linkedin.com/in/vemurivenkata-satya-markandeyapavanateja-0651a0403)
- Email: pavanatejavemuri2007@gmail.com

---

## 📄 License

This repository does not currently specify a license. Consider adding an [MIT License](https://choosealicense.com/licenses/mit/) if you'd like others to freely reuse or build on these practicals.

---

⭐ If you find these practicals useful for learning Deep Learning fundamentals, consider giving the repo a star!
