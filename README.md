# Incremental Learning on GTSRB with EWC and Replay

## Overview
This project explores **class-incremental learning** for the **German Traffic Sign Recognition Benchmark (GTSRB)**.  
We investigate different strategies to prevent **catastrophic forgetting** when adding new traffic sign classes sequentially.

The implemented approaches include:
- **Elastic Weight Consolidation (EWC)** – Regularizes the model to preserve important parameters from previous tasks.
- **Replay (Memory Buffer)** – Stores past samples and reuses them during training to retain knowledge.
- **Replay + EWC (Hybrid)** – A combination of weight regularization and experience replay for improved performance.

The repository allows testing **EWC, Replay, and their combination** to compare their effectiveness in incremental learning.

## Problem Statement
Incremental learning presents a challenge: **how can a model learn new classes without forgetting old ones?**  
Traditional neural networks suffer from **catastrophic forgetting**, meaning that new knowledge **overwrites previous knowledge** unless specific strategies are used.

This project follows **a structured approach** to incremental learning, using:
- **Rehearsal (Memory Buffer)** – Stores past task samples and mixes them with new training data.
- **EWC (Elastic Weight Consolidation)** – Regularizes the network to protect key weights from previous tasks.
- **Replay + EWC** – Merges both strategies for stronger memory retention.

## Strategy Adopted

### **1️⃣ Dataset: GTSRB**
- The dataset is **automatically downloaded** using `torchvision.datasets.GTSRB`.
- Images are **preprocessed (resized, normalized) and loaded dynamically**.
- Tasks are defined to incrementally introduce new traffic sign classes.

### **2️⃣ Incremental Learning Methods**
We test and compare three approaches:

#### **✔️ EWC (Elastic Weight Consolidation)**
- Computes **Fisher Information Matrix (FIM)** to estimate parameter importance.
- Applies **regularization loss** to prevent modifying critical weights.
- Helps retain old knowledge **without storing old data**.

#### **✔️ Replay (Experience Replay)**
- Maintains a **memory buffer** of past task samples.
- Replays stored samples during training to prevent forgetting.
- Uses a **FIFO-based buffer** to manage memory constraints.

#### **✔️ Replay + EWC (Hybrid)**
- Integrates **both EWC and Replay** to maximize knowledge retention.
- Uses **EWC to protect model parameters** while **Replay reintroduces old data**.

### **3️⃣ Training Process**
- **New classes** are introduced incrementally in multiple tasks.
- A **pretrained model** is first trained on an initial set of classes.
- **EWC, Replay, or Replay + EWC** is applied to retain old knowledge.
- The model is evaluated on **all encountered classes after each task**.

## Results

| **Task** | **EWC Accuracy (%)** | **Replay Accuracy (%)** | **Replay + EWC (%)** |
|----------|---------------------|----------------------|--------------------|
| Task 1   | XX.XX% | XX.XX% | XX.XX% |
| Task 2   | XX.XX% | XX.XX% | XX.XX% |
| Task 3   | XX.XX% | XX.XX% | XX.XX% |
| Task 4   | XX.XX% | XX.XX% | XX.XX% |

_(Results will be updated as experiments progress)_

## How to Run

### **1️⃣ Run in Google Colab**
Click the button below to **open the notebook in Colab**:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13LKChqohIFDFjcDnop7lbiNu8p1yqOiV?usp=sharing)

### **2️⃣ Run Locally**
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Incremental-Learning-GTSRB.git
   cd Incremental-Learning-GTSRB

### Future Work
- Explore additional regularization techniques.
- Compare Replay + EWC with more advanced lifelong learning strategies.
- Optimize buffer management for better memory efficiency.

### Acknowledgements & References  

This project is based on well-established research in **incremental learning and catastrophic forgetting**.  
The following references provide the theoretical foundation for the implemented methods:

- **Elastic Weight Consolidation (EWC)**
  - Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al.  
    ["Overcoming Catastrophic Forgetting in Neural Networks."](https://www.pnas.org/doi/10.1073/pnas.1611835114)  
    *Proceedings of the National Academy of Sciences (PNAS), 2017.*

- **Experience Replay (Memory Buffer)**
  - Rebuffi, S. A., Kolesnikov, A., Sperl, G., & Lampert, C. H.  
    ["iCaRL: Incremental Classifier and Representation Learning."](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html)  
    *CVPR, 2017.*

- **Replay + EWC (Hybrid Approach)** 
  - De Lange, M., Aljundi, R., Masana, M., et al.  
    ["Deep Continual Learning: A Comprehensive Review."](https://doi.org/10.1016/j.neunet.2021.12.002)  
    *Neural Networks, 2022.*

