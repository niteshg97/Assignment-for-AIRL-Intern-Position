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

### Observations:

* Initial training without strong augmentation resulted in ~50% accuracy due to overfitting.
* Augmentations such as **RandomResizedCrop**, **RandomHorizontalFlip**, and **TrivialAugmentWide** were crucial to improve generalization.
* Simply increasing depth (e.g., 12 layers with a small embedding) was ineffective.
* Increasing model **width (embed_dim=192)** led to better feature representation and higher accuracy.

---

## 🔑 Key Takeaways

* **Data augmentation** is critical when training ViTs on small datasets like CIFAR-10.
* **Balanced model design** (moderate depth, sufficient width) performs better than deeper but narrow networks.
* Regularization and learning rate scheduling (CosineAnnealingLR) stabilize training and boost performance.

---

## 📌 File Structure

* **q1.ipynb** – Complete notebook containing implementation, training, and evaluation of the Vision Transformer.

---

## 🏆 Conclusion

This project highlights how **careful architectural design, regularization, and augmentations** make it possible to train a Vision Transformer effectively on small-scale datasets such as CIFAR-10.
