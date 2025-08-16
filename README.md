# 📑 Multi-Label News Classification using BERT

This project implements a **multi-label text classification system** on the **Reuters-21578 dataset** using **BERT** and **PyTorch**.  
Each news article may belong to multiple categories (e.g., *earn, trade, grain*), making this a real-world multi-label NLP task.

---

## 📂 Dataset: Reuters-21578
- The **Reuters-21578 dataset** is a collection of **21,578 news documents** from Reuters in 1987.  
- Articles are categorized into **90+ overlapping categories**, allowing **multi-label classification**.  
- For this project, we use the CSV version of the dataset where:
  - **`text`** column → contains the news article content.  
  - **`labels`** column → contains one or more categories per article (comma-separated).  

Example row:

| text | labels |
|------|---------|
| "The U.S. trade deficit narrowed in September..." | trade, money-fx |

---

## ⚙️ Project Workflow
1. **Data Preprocessing**
   - Load the CSV dataset
   - Clean and tokenize text
   - Convert labels to **multi-hot vectors**  

2. **Model**
   - Fine-tuned **BERT (bert-base-uncased)** for text classification  
   - Added a **sigmoid output layer** for multi-label prediction  

3. **Training**
   - Optimizer: AdamW  
   - Loss Function: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)  
   - Batch Size: 16  
   - Epochs: 5  

4. **Evaluation**
   - Metrics: **Micro-F1, Macro-F1, Precision, Recall**  
   - Handles **imbalanced label distribution**  

---

## 📊 Results
- **Micro-F1:** ~92%  
- **Macro-F1:** ~89%  
- Optimized inference speed by **35%** using batch processing  

---
