# Incremental Learning on GTSRB with EWC and Replay

## Overview
This project explores **class-incremental learning** for the **German Traffic Sign Recognition Benchmark (GTSRB)**.  
We use **Elastic Weight Consolidation (EWC) and Replay (Memory Buffer)** to prevent **catastrophic forgetting** when adding new traffic sign classes sequentially.

## Problem Statement
Incremental learning presents a challenge: **how can a model learn new classes without forgetting old ones?**  
Traditional neural networks suffer from **catastrophic forgetting**, meaning that new knowledge **overwrites previous knowledge** unless specific strategies are used.  

This project follows **a more conventional approach** (compared to Mixture-of-Experts) using:
- **Rehearsal (Memory Buffer)** 🗂️ – Stores past samples to mix with new training data.
- **EWC (Elastic Weight Consolidation)** 🔗 – Regularizes the network by preserving important weights from previous tasks.

## Strategy Adopted

### **1️⃣ Dataset: GTSRB**
- The dataset is **automatically downloaded** using `torchvision.datasets.GTSRB`.
- Images are **preprocessed and loaded dynamically**.

### **2️⃣ Training Approach**
- We introduce **new classes incrementally** in multiple tasks.
- A **pretrained model** is first trained on the initial task.
- We apply **Replay + EWC** to maintain old knowledge.

### **3️⃣ Implementation Details**
- **Backbone Model:** Uses a **LeNet-based CNN**.
- **Training Strategy:** Each task is trained sequentially with Replay and EWC.
- **Evaluation:** Performance is tested on **all seen classes** after each task.

## Results

| **Task** | **Accuracy (%)** |
|----------|----------------|
| Task 1   | XX.XX% |
| Task 2   | XX.XX% |
| Task 3   | XX.XX% |
| Task 4   | XX.XX% |

_(Results will be updated as experiments progress)_

## How to Run

### **1️⃣ Run in Google Colab**
Click the button below to **open the notebook in Colab**:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](COLAB_LINK_HERE)

### **2️⃣ Run Locally**
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Incremental-Learning-GTSRB.git
   cd Incremental-Learning-GTSRB

### Future Work
- Test additional regularization strategies
- Compare Replay + EWC with other techniques
- 

### Acknowledgements
- This project is based on the GTSRB dataset and follows incremental learning best practices.
- Inspired by NeurIPS Tutorials on Lifelong Learning.
