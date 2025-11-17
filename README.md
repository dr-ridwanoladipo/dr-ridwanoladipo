<div align="center">

# **Ridwan Oladipo, MD**

### Medical Data Scientist • Clinical AI Architect • MLOps Engineer

*Building end-to-end medical AI — **7+ production deployments** (neurosurgery, cardiology, radiology, pharmacology) on **AWS** with **<200ms inference** and **full CI/CD automation***

**Deep Learning • RAG • LLMs • MLOps • Explainable AI (SHAP/Grad-CAM)**

**Stanford • Harvard • Duke • Oxford • Johns Hopkins**

---

[![Portfolio](https://img.shields.io/badge/🌐_PORTFOLIO-MEDNEXAI.COM-1e3c72?style=for-the-badge)](https://mednexai.com)
[![LinkedIn](https://img.shields.io/badge/LINKEDIN-CONNECT-0077b5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/drridwanoladipoai)
[![Email](https://img.shields.io/badge/EMAIL-CONTACT-d14836?style=for-the-badge&logo=gmail)](mailto:dr.ridwan.oladipo@gmail.com)

</div>

---

## 🎯 Executive Overview

MD delivering **production-grade clinical AI systems** across **cardiology, ECG diagnostics, neurosurgery, radiology (MRI), and pharmacology** — achieving **97% cardiovascular sensitivity**, **96% MI detection**, **86% tumor-segmentation Dice in 5.8s**, and **170K+ RAG-powered drug-interaction coverage** at **<200ms latency**.

Each solution unifies **deep learning, RAG, and LLMs** with **SHAP/Grad-CAM explainability**, **FastAPI backends**, and **AWS Fargate CI/CD** for clinically safe, enterprise-ready deployment.

---

## 🚀 Flagship Projects


### 🌐 Deployment Options
- **Live Demos:** Instant access via HuggingFace (UI + API)  
- **Production (On-Demand):** Fully deployed on AWS ECS Fargate — available by request  
> ⚡ **AWS Production:** CI/CD-enabled with <10 minutes cold-start (cost-optimized)

---

### 🩺 **Heart Disease Risk AI**
**Problem:** Heart disease causes 17.9 million deaths worldwide annually; manual ASCVD scoring delays emergency triage  
**Performance:** **97% sensitivity** · **0.91 AUC** · **<200ms inference** · **SHAP explainability**  
**Deployment:** XGBoost + FastAPI + Streamlit → AWS Fargate CI/CD  

[![🎬 UI Demo](https://img.shields.io/badge/🎬_UI_Demo-Live-blue?style=flat-square)](https://huggingface.co/spaces/dr-ridwanoladipo/cardio-ai)
[![🔗 API Demo](https://img.shields.io/badge/🔗_API_Demo-Live-green?style=flat-square)](https://huggingface.co/spaces/dr-ridwanoladipo/cardio-ai-api)  
[![🚀 Production (AWS ECS Fargate)](https://img.shields.io/badge/🚀_Production-cardio.mednexai.com-f59e0b?style=flat-square)](#-deployment-options)    
[![GitHub](https://img.shields.io/badge/Code-Repository-00aa00?style=flat&logo=github&logoColor=white)](https://github.com/dr-ridwanoladipo/cardio-ai)
---

### 🫀 **Clinical-Grade ECG Diagnosis AI**
**Problem:** Delayed ECG interpretation increases MI mortality; limited cardiologist availability creates critical bottlenecks in emergency workflows  
**Performance:** **96.2% MI sensitivity** · **99.9% specificity** · **0.9986 AUC** · **Grad-CAM + SHAP**  
**Deployment:** ResNet-1D + TensorFlow → FastAPI → AWS Fargate with zero-downtime rollback  

[![🎬 UI Demo](https://img.shields.io/badge/🎬_UI_Demo-Live-blue?style=flat-square)](https://huggingface.co/spaces/dr-ridwanoladipo/ecg-ai)
[![🔗 API Demo](https://img.shields.io/badge/🔗_API_Demo-Live-green?style=flat-square)](https://huggingface.co/spaces/dr-ridwanoladipo/ecg-ai-api)  
[![🚀 Production (AWS ECS Fargate)](https://img.shields.io/badge/🚀_Production-ecg.mednexai.com-f59e0b?style=flat-square)](#-deployment-options)  
[![GitHub](https://img.shields.io/badge/Code-Repository-00aa00?style=flat&logo=github&logoColor=white)](https://github.com/dr-ridwanoladipo/ecg-ai)

---

### 🧠 **Brain Tumor Segmentation AI**
**Problem:** Manual MRI tumor segmentation takes 25–45 minutes per case, delaying critical neurosurgical decisions   
**Performance:** **86% Whole-Tumor Dice** · **5.8s inference** · **nnU-Net 2025** · **Robustness-tested**  
**Deployment:** PyTorch + SageMaker training → FastAPI + Streamlit → AWS Fargate with PACS-ready DICOM workflow  

[![🎬 UI Demo](https://img.shields.io/badge/🎬_UI_Demo-Live-blue?style=flat-square)](https://huggingface.co/spaces/dr-ridwanoladipo/brain-tumor-ai)
[![🔗 API Demo](https://img.shields.io/badge/🔗_API_Demo-Live-green?style=flat-square)](https://huggingface.co/spaces/dr-ridwanoladipo/brain-tumor-api)  
[![🚀 Production Ready (AWS ECS Fargate)](https://img.shields.io/badge/🚀_Production-brain.mednexai.com-f59e0b?style=flat-square)](#-deployment-options)  
[![GitHub](https://img.shields.io/badge/Code-Repository-00aa00?style=flat&logo=github&logoColor=white)](https://github.com/dr-ridwanoladipo/brain-tumor-ai)

---

### 💊 **Drug Interaction AI**
**Problem:** 1.3M+ annual U.S. ER visits from adverse drug events; pharmacist reviews take 15–20 min  
**Performance:** **170K+ DrugBank interactions** · **<200ms KB lookup** · **GPT-5 RAG + FAISS semantic retrieval** · **89.8% RxNorm normalization**  
**Deployment:** Tier-adaptive retrieval (direct KB → FAISS → GPT-5) → FastAPI rate-limited endpoints → AWS Fargate  

[![🎬 UI Demo](https://img.shields.io/badge/🎬_UI_Demo-Live-blue?style=flat-square)](https://huggingface.co/spaces/dr-ridwanoladipo/drug-interaction-ai) 
[![🔗 API Demo](https://img.shields.io/badge/🔗_API_Demo-Live-green?style=flat-square)](https://huggingface.co/spaces/dr-ridwanoladipo/drug-interaction-api)  
[![🚀 Production (AWS Fargate)](https://img.shields.io/badge/🚀_Production-drug.mednexai.com-f59e0b?style=flat-square)](#-deployment-options)  
[![GitHub](https://img.shields.io/badge/Code-Repository-00aa00?style=flat&logo=github&logoColor=white)](https://github.com/dr-ridwanoladipo/drug-interaction-ai)

---

## 🏗️ Unified Technical Architecture

**All systems follow a consistent production pattern** — demonstrating repeatable MLOps engineering across deep learning, classical ML, and LLM-based retrieval:
```mermaid
graph LR
    A[Clinical Input<br/>Patient Data/ECG/MRI/Drugs] --> B[Streamlit UI<br/>Clinical Interface]
    
    B --> C[FastAPI Backend<br/>Validation · Rate Limiting · Health Checks]
    
    C --> D{Model Layer}
    
    D -->|Cardio| E1[XGBoost + SHAP]
    D -->|ECG| E2[ResNet-1D + Grad-CAM]
    D -->|Brain| E3[nnU-Net 2025]
    D -->|Drug| E4[FAISS + GPT-5 RAG]
    
    E1 --> F[Prediction + Explainability]
    E2 --> F
    E3 --> F
    E4 --> F
    
    F --> G[Docker Container<br/>Multi-stage Build]
    
    G --> H[AWS ECS Fargate<br/>Auto-scaling · Load Balancer]
    
    H --> I[GitHub Actions CI/CD<br/>Automated Deploy · Rollback]
    
    I --> J[CloudWatch Monitoring<br/>Logs · Metrics · Alerts]
    
    subgraph "Clinical Interface Layer"
        A
        B
    end
    
    subgraph "API & Validation Layer"
        C
    end
    
    subgraph "AI Model Layer"
        D
        E1
        E2
        E3
        E4
        F
    end
    
    subgraph "Production Infrastructure"
        G
        H
        I
        J
    end
    
    style A fill:#e1f5fe
    style D fill:#fff3e0
    style H fill:#f3e5f5
    style I fill:#e8f5e9
```

**Stack Consistency Across All Projects:**  
✅ **Frontend:** Streamlit clinical UI with domain-specific visualizations  
✅ **Backend:** FastAPI with Pydantic validation, rate limiting (SlowAPI), Swagger docs  
✅ **Deployment:** Docker multi-stage builds → AWS ECS Fargate → Application Load Balancer  
✅ **CI/CD:** GitHub Actions automated pipelines (~5 min git push → production)  
✅ **Monitoring:** CloudWatch logs, health checks, automated rollback on failure  
✅ **Explainability:** SHAP (tabular), Grad-CAM (imaging), tier-stamped confidence (RAG)

---

## 📈 Clinical Impact Summary

| Project | Clinical Value | Business ROI | Enterprise Readiness |
|---------|----------------|--------------|---------------------|
| **Heart Disease Risk AI** | Automates ASCVD/Framingham scoring · Identifies high-risk patients before acute events · Reduces avoidable ED admissions | **8,000+ physician-hours saved annually** (500-bed hospital) · Preventive care cost avoidance | **EHR-ready API** · Epic/Cerner integration · <200ms real-time response |
| **ECG Diagnosis AI** | Accelerates MI triage · Standardizes interpretation in rural/understaffed settings · Reduces door-to-balloon time | **$1.3M+ annual litigation risk prevented** per hospital · Fewer missed diagnoses | **PACS-compatible** · Zero false-negative tolerance · Multi-site PTB-XL validation |
| **Brain Tumor Segmentation AI** | Cuts segmentation time from 25–45 min → **5.8 seconds** (99% faster) · Surgical-grade boundary precision for resection planning | **Radiologist bottleneck elimination** · Faster treatment decisions · Radiotherapy workflow acceleration | **DICOM-native** · PACS integration · nnU-Net clinical-grade standard |
| **Drug Interaction AI** | Automates 15–20 min pharmacist reviews · Real-time polypharmacy risk synthesis · Catches brand-name mismatches (89.8% RxNorm success) | **8,000+ hours saved annually** · **$50K+ per ADE prevented** (litigation + LOS) | **Epic/Cerner CPOE pluggable** · <200ms Tier-1 lookup · FDA SaMD-aligned safety logic |

___

## 🔬 Modeling & Explainability Suite

- **Modeling:** XGBoost (tabular cardiology), ResNet-1D (ECG waveforms), nnU-Net 2025 (MRI), RAG + GPT-5 (pharmacology)  
- **Explainability:** SHAP (tabular), Grad-CAM (ECG), 3D segmentation overlays (MRI), tiered evidence reasoning (RAG)  
- **Outcome:** Clinically transparent, auditable AI aligned with FDA SaMD expectations — never black-box models

---

## 🛡️ Safety, Compliance & Validation

- **Regulatory alignment:** Built under **FDA SaMD principles** with stratified splits and robustness checks (noise, intensity shifts, demographic subgroups)  
- **Clinical safeguards:** All deployments include medical disclaimers — AI assists clinicians, never replaces judgment  
- **Operational safety:** Tiered confidence logic (Drug AI), clinical plausibility checks (Cardio), high-sensitivity prioritization (ECG/Brain)

---

## 🛠️ Tech Stack

**Deep Learning & ML**  
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

**LLMs & Retrieval**  
![OpenAI](https://img.shields.io/badge/OpenAI_GPT--5-412991?style=flat&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat&logo=meta&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white)

**MLOps & Deployment**  
![AWS](https://img.shields.io/badge/AWS_Fargate-FF9900?style=flat&logo=amazon-aws&logoColor=white)
![SageMaker](https://img.shields.io/badge/AWS_SageMaker-232F3E?style=flat&logo=amazonaws&logoColor=white)
![Bedrock](https://img.shields.io/badge/AWS_Bedrock-FF9900?style=flat&logo=amazonaws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=github-actions&logoColor=white)

**Frontend & Explainability**  
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-00599C?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)

**Clinical Data & Standards**  
![DICOM](https://img.shields.io/badge/DICOM-005571?style=flat)
![RxNorm](https://img.shields.io/badge/RxNorm-4A148C?style=flat)
![DrugBank](https://img.shields.io/badge/DrugBank-2E7D32?style=flat)

---

## 🎓 Professional Training

**Stanford University** — AI in Healthcare Specialization (5 courses)  
**University of Oxford** — Generative and Agentic AI  
**Harvard University** — CS50x: Introduction to Computer Science  
**Harvard University** — Machine Learning and AI with Python  
**Duke University** — MLOps: Machine Learning Operations Specialization (4 courses)  
**Johns Hopkins University** — Generative AI in Public Health

---

## 🤝 Collaboration & Contact

[![🌐 portfolio](https://img.shields.io/badge/🌐_portfolio-mednexai.com-1e3c72?style=flat-square)](https://mednexai.com)
[![linkedin](https://img.shields.io/badge/linkedin-connect-0077b5?style=flat-square&logo=linkedin)](https://linkedin.com/in/drridwanoladipoai)
[![email](https://img.shields.io/badge/email-contact-d14836?style=flat-square&logo=gmail)](mailto:dr.ridwan.oladipo@gmail.com)

**Open to:** Medical Data Scientist · Clinical AI Architect · Applied ML/MLOps Engineer  
**Collaboration:** Hospitals, AI startups, research labs, telemedicine companies, and engineering teams building real-world medical AI products

---
