# Project_507
# Crisis Severity Detection with TF–IDF Logistic Regression and DistilBERT: 
*A Cost-Sensitive Approach with Lightweight Defense-Oriented Adaptation

## Report
- 📄 [Crisis Severity Detection Project Report (PDF)](Project_507_Donghyun_Kim.pdf)

---
## 📘 Overview
This project implements a complete pipeline for crisis severity detection using the CrisisMMD humanitarian dataset.  
The workflow includes:

- Construction of a **4-level operational severity scale**
- A **TF–IDF + Logistic Regression** baseline model
- **DistilBERT** fine-tuning for contextual classification
- Lightweight **lexicon-guided domain adaptation**
- A custom **cost-sensitive evaluation framework** to capture asymmetric operational risks

The goal is to examine how classical and Transformer-based models differ in handling crisis-related text, and whether simple domain adaptation can improve performance in defense-relevant contexts.

## 📊 Dataset
**CrisisMMD (Humanitarian subset)**  
- Source: Public dataset released by Alam et al. (ICWSM 2018)  
- Contains short, noisy Twitter messages from real disaster events  
- Eight original humanitarian labels → mapped to four severity levels:
  - **Routine**  
  - **Elevated**  
  - **Urgent**  
  - **Critical**

The mapping is inspired by:
- U.S. Army medical evacuation precedence (ATP 4-02.2)  
- Emergency Severity Index (ESI) clinical triage system  

---

## 🧭 Method Summary

### **1. Exploratory Data Analysis**
- Heavy class imbalance  
- High lexical noise  
- Short tweets (~15 words on average)  
- Rare but operationally important classes (e.g., Critical)

### **2. Severity Mapping**
Eight original CrisisMMD humanitarian categories are grouped into four levels:

| Severity | Original Labels |
|---------|-----------------|
| Routine | not_humanitarian, other_relevant_information |
| Elevated | infrastructure_and_utility_damage, vehicle_damage, rescue_volunteering_or_donation_effort |
| Urgent | affected_individuals |
| Critical | injured_or_dead_people, missing_or_found_people |

This produces an ordered, decision-oriented target variable.

### **3. TF–IDF + Logistic Regression**
- Unigrams + bigrams  
- Vocabulary capped at 10,000  
- lbfgs optimizer  
- Class weights = inverse-frequency  

Serves as a classical, non-contextual baseline.

### **4. DistilBERT Baseline**
- Max sequence length = 64  
- Weight decay 0.01  
- Batch size 16/32  
- 3 epochs  
Fine-tuned as a four-class classifier.

### **5. Domain-Adapted DistilBERT**
- Custom defense logistics lexicon from:
  - NATO AJP-4  
  - JP 4-0  
  - AJP-4.4  
  - Humanitarian logistics literature
- Tweets containing lexicon terms oversampled  
- No model architecture changes  
- Goal: expose DistilBERT to defense-related contexts

### **6. Cost-Sensitive Evaluation**
A 4×4 custom risk matrix penalizes:
- Underestimating **Urgent/Critical** most severely  
- Severe false alarms
- Large severity distance  

Outputs:
- Total risk  
- Average risk  

---

## 🧪 Key Results

### **Baseline Comparison**
| Model | Accuracy | Macro-F1 | Avg Risk |
|-------|----------|----------|----------|
| Logistic Regression | 0.646 | 0.472 | 0.585 |
| DistilBERT | 0.703 | 0.522 | 0.421 |

➡ DistilBERT clearly outperforms TF–IDF baseline.

### **Domain Adaptation Effect**
| Model | Accuracy | Macro-F1 | Avg Risk |
|-------|----------|----------|----------|
| DistilBERT | 0.703 | 0.522 | 0.421 |
| Domain-Adapted DistilBERT | 0.685 | 0.476 | 0.461 |

➡ Oversampling using the defense lexicon shifts model behavior  
➡ …but **does NOT improve performance**

Key observations:
- More Routine → Elevated misclassifications  
- Critical recall drops (0.67 → 0.41)  
- Model becomes more sensitive to logistics-related cues  
- But this hurts generalization on the CrisisMMD test distribution

---

## 🏁 Conclusion
- DistilBERT improves crisis-severity classification over TF–IDF.  
- Both models struggle with rare high-severity cases (Urgent/Critical).  
- Simple lexicon-weighted oversampling is **not** enough for domain specialization.  
- More robust adaptation (sample weighting, continual pretraining) is needed.  
- Public crisis datasets are effective proxies for developing triage-style classifiers.

---

## 🔗 References  
(References already included in the notebook and final report.)


- Gilboy et al., *Emergency Severity Index (ESI)*  
- U.S. Army, ATP 4-02.2: *Medical Evacuation*  
- NATO AJP-4, AJP-4.4  
- JP 4-0: *Joint Logistics*  
- Tatham & Houghton, *Humanitarian Logistics*  
- Wolf et al., *DistilBERT (Transformers)*

---

## 📫 Contact
**Donghyun Kim**  
Department of Statistics  
University of Michigan  
📧 donghki@umich.edu  

---
