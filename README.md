# Incremental Learning on GTSRB with EWC and Replay

## Overview
This project explores **class-incremental learning** for the **German Traffic Sign Recognition Benchmark (GTSRB)**.  
We implement and compare strategies to mitigate **catastrophic forgetting** when introducing new traffic sign classes sequentially.

## Problem Statement & Approach
Neural networks tend to **overwrite previous knowledge** when learning new classes, a phenomenon known as **catastrophic forgetting**.  
To address this, we evaluate the following methods:

- **Elastic Weight Consolidation (EWC)** – Regularizes key model parameters to retain prior knowledge.
- **Replay (Memory Buffer)** – Stores and reuses past samples to reinforce learning.
- **Replay + EWC (Hybrid)** – Combines regularization with experience replay for stronger memory retention.

## Strategy Adopted

### **1️⃣ Dataset: GTSRB**
- Automatically downloaded via `torchvision.datasets.GTSRB`.
- Images are **preprocessed (resized, normalized) and dynamically loaded**.
- New classes are introduced incrementally.

### **2️⃣ Incremental Learning Methods**
We compare three approaches:

#### **✔️ EWC**
- Estimates parameter importance using the **Fisher Information Matrix (FIM)**.
- Applies **regularization loss** to preserve critical weights.
- Works without storing past data.

#### **✔️ Replay**
- Stores and replays past samples to prevent forgetting.
- Uses a **FIFO-based buffer** to manage memory constraints.

#### **✔️ Replay + EWC**
- Integrates **both EWC and Replay** for optimal retention.
- Protects parameters while reinforcing learning with past data.

### **3️⃣ Training Process**
- **Classes are introduced incrementally** over multiple tasks.
- A **pretrained model** learns an initial class set.
- **EWC, Replay, or Replay + EWC** is applied to retain past knowledge.
- The model is **evaluated on all encountered classes after each task**.

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
  - Robins, A.  
    ["Catastrophic Forgetting, Rehearsal, and Pseudorehearsal."](https://doi.org/10.1016/0893-6080(95)00026-6)  
    *Neural Networks, 1995.*
    
- **Replay + EWC (Hybrid Approach)** 
  - De Lange, M., Aljundi, R., Masana, M., et al.  
    ["Deep Continual Learning: A Comprehensive Review."](https://doi.org/10.1016/j.neunet.2021.12.002)  
    *Neural Networks, 2022.*

