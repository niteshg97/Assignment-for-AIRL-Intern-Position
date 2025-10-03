# Vision Transformer (ViT) on CIFAR-10

## 📌 Goal

The objective of this project is to implement a **Vision Transformer (ViT)** from scratch using **PyTorch** and train it on the **CIFAR-10 dataset (10 classes)**. The design and training process are inspired by the paper *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*.

---

## 📂 Project Description

This repository contains the implementation of a Vision Transformer tailored for CIFAR-10. The focus was to:

* Build a ViT from scratch.
* Train it effectively on a relatively small dataset.
* Achieve the best possible accuracy through model scaling, regularization, and data augmentation.

---

## ▶️ How to Run the Code

All code is contained in **q1.ipynb**, optimized for running on **Google Colab**.

1. **Open in Colab**

   * Either click the *"Open in Colab"* button on GitHub or manually upload `q1.ipynb` to Colab.
2. **Select GPU Runtime**

   * Navigate to *Runtime → Change runtime type → GPU*.
3. **Run All Cells**

   * Execute all cells to train and evaluate the model.

---

## ⚙️ Hyperparameters

| Parameter                  | Value             |
| -------------------------- | ----------------- |
| Image Size                 | 32×32             |
| Patch Size                 | 4×4               |
| Batch Size                 | 128               |
| Embedding Dimension        | 192               |
| MLP Dimension              | 384               |
| Transformer Layers (Depth) | 6                 |
| Attention Heads            | 4                 |
| Optimizer                  | AdamW             |
| Learning Rate              | 0.0003            |
| Weight Decay               | 0.05              |
| LR Scheduler               | CosineAnnealingLR |
| Epochs                     | 50                |
| Dropout Rate               | 0.1               |

---

## 📊 Results

* **Final Test Accuracy:** **77.90%** (after 50 epochs on CIFAR-10).

The model was trained for 50 epochs, and the final test accuracy was measured on the unseen CIFAR-10 test set.

**Test Accuracy: 77.90%**

My initial attempts without significant data augmentation barely crossed the 50% accuracy mark, as the model quickly began to overfit. Introducing augmentations like RandomResizedCrop, RandomHorizontalFlip, and especially TrivialAugmentWide was the single most impactful change. It forces the model to learn the underlying features of an object rather than just memorizing pixel patterns, which is essential for generalization.

For a small 32x32 image, simply making the model deeper (e.g., 12 transformer layers) with a tiny embedding dimension didn't yield good results. A better strategy was to make the model moderately "wide" by increasing the embed_dim to 192. This gives each layer more capacity to create rich feature representations from the patches, which proved more effective than passing a low-information vector through many layers.


## 📌 File Structure

* **q1.ipynb** – Complete notebook containing implementation, training, and evaluation of the Vision Transformer.

---

## 🏆 Conclusion

This project highlights how **careful architectural design, regularization, and augmentations** make it possible to train a Vision Transformer effectively on small-scale datasets such as CIFAR-10.
