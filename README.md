# BERT-for-RE: Aspect-Based Sentiment Analysis for Requirements Elicitation
This repository contains the implementation of **Aspect-Based Sentiment Analysis (ABSA)** using a **fine-tuned BERT** model with integrated **Explainable Artificial Intelligence (XAI)** techniques such as **LIME**. The work supports automating the **Requirement Elicitation (RE)** process using app reviews and was conducted on the **AWARE dataset**.

This implementation is based on the paper:  
**"Aspect-Based Sentiment Analysis for Software Requirements Elicitation using Fine-Tuned BERT and Explainable AI"**  
> **Authors**: Soonh Taj, Sher Muhammad Daudpota, Ali Shariq Imran, Zenun Kastrati  

---

## **Table of Contents**  
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Requirements](#requirements)  
- [Model Architecture](#model-architecture)  
- [How to Run the Code](#how-to-run-the-code)  
- [Results](#results)  
- [Explainability with LIME](#explainability-with-lime)  
- [Citation](#citation)  
- [License](#license)  

---

## **Overview**  
This repository automates the analysis of app reviews to extract user requirements using Aspect-Based Sentiment Analysis (ABSA). Key highlights include:  
1. Fine-tuning of **BERT** for ABSA tasks:  
   - Aspect Category Detection (ACD)  
   - Aspect Category Polarity (ACP)  
2. Integration of **LIME** (Local Interpretable Model-Agnostic Explanations) to explain predictions.  
3. Experiments conducted on the **AWARE dataset**, achieving state-of-the-art results.  

---

## **Dataset**  
The **AWARE dataset** (ABSA Warehouse of Apps REviews) is specifically designed for Requirement Elicitation (RE). It includes 11,323 app reviews across three domains:  
- **Social Networking**  
- **Productivity**  
- **Games**  

Download the dataset: [AWARE Dataset](#) (Add download link if available).

---

## **Requirements**  
To set up the environment, install the following libraries:  
```bash
pip install torch transformers scikit-learn lime pandas numpy matplotlib optuna
```  

### **Python Version**  
- Python 3.8+  

---

## **Model Architecture**  
- **BERT (Bidirectional Encoder Representations from Transformers)**: Pre-trained BERT model fine-tuned for ACD and ACP tasks.  
- **LIME**: Post-hoc explanations for interpreting BERT predictions at a local level.  

The architecture includes:  
1. Text preprocessing and tokenization using BERT tokenizer.  
2. Fine-tuning of BERT for ABSA tasks.  
3. Post-hoc explanations using LIME.

---

## **How to Run the Code**  

### 1. Clone the Repository  
```bash
git clone https://github.com/engrsoonhtaj/BERT-for-RE.git
cd BERT-for-RE
```

### 2. Prepare the Dataset  
Ensure the **AWARE dataset** is placed in the `data/` directory:  
```bash
data/
    └── aware_dataset.csv
```

### 3. Train the Model  
Run the following command to fine-tune BERT:  
```bash
python train.py --epochs 16 --batch_size 8 --learning_rate 1e-5
```

### 4. Evaluate the Model  
Evaluate the model on the test set:  
```bash
python evaluate.py --checkpoint_path checkpoints/best_model.pt
```

### 5. Generate Explanations with LIME  
Run the following to generate LIME-based explanations:  
```bash
python lime_explainer.py --input_text "This app has bugs after the update."
```

---

## **Results**  

### **Performance Metrics**  
| Domain              | ACD (F1-Score) | ACP (Accuracy) |  
|---------------------|----------------|----------------|  
| Social Networking   | 78%           | 92%           |  
| Productivity        | 82%           | 92%           |  
| Games               | 89%           | 96%           |  

Our model outperforms baseline methods such as SVM, CNN, and previous BERT-based approaches.

---

## **Explainability with LIME**  
LIME generates local explanations for BERT predictions. For example:  

**Input Review**: "This app crashes frequently after the update."  
**Predicted Aspect**: *Reliability*  
**Polarity**: *Negative*  

**LIME Explanation**:  
- Tokens like "crashes" and "frequently" contributed most to the *Negative* polarity prediction.  

To visualize LIME explanations, run the script `lime_explainer.py` with any input text.

---

## **Citation**  
If you use this code in your work, please cite the following:  

```bibtex
@article{taj2024absa,
  title={Aspect-Based Sentiment Analysis for Software Requirements Elicitation using Fine-Tuned BERT and Explainable AI},
  author={Soonh Taj, Sher Muhammad Daudpota, Ali Shariq Imran, Zenun Kastrati},
  journal={Engineering Applications of Artificial Intelligence},
  year={2024}
}
```

---

## **License**  
This project is licensed under the MIT License.  

---
