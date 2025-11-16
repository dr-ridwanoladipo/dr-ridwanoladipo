<div align="center">

# **Ridwan Oladipo, MD**

### Medical Data Scientist • Clinical AI Architect • MLOps Engineer

*Building end-to-end medical AI — **7+ production deployments** (neurosurgery, cardiology, radiology, pharmacology) on **AWS** with **<200ms inference** and **full CI/CD automation***

**Deep Learning • RAG • LLMs • MLOps • Explainable AI (SHAP/Grad-CAM)**

**Stanford • Harvard • Duke • Oxford • Johns Hopkins**

---

[![Portfolio](https://img.shields.io/badge/🌐_PORTFOLIO-MEDNEXAI.COM-1e3c72?style=for-the-badge)](https://mednexai.com)
[![LinkedIn](https://img.shields.io/badge/LINKEDIN-CONNECT-0077b5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/drridwanoladipoai)
[![Email](https://img.shields.io/badge/EMAIL-COLLABORATE-d14836?style=for-the-badge&logo=gmail)](mailto:dr.ridwan.oladipo@gmail.com)

</div>

---

## 🎯 Executive Overview

MD delivering **production-grade clinical AI systems** across **cardiology, ECG diagnostics, neurosurgery, radiology (MRI), and pharmacology** — achieving **97% cardiovascular sensitivity**, **96% MI detection**, **86% tumor-segmentation Dice in 5.8s**, and **170K+ RAG-powered drug-interaction coverage** at **<200ms latency**.

Each solution unifies **deep learning, RAG, and LLMs** with **SHAP/Grad-CAM explainability**, **FastAPI backends**, and **AWS Fargate CI/CD** for clinically safe, enterprise-ready deployment.

---

## 🚀 Flagship Projects

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
**Performance:** **96.2% MI sensitivity** · **99.9% specificity** · **0.999 AUC** · **Grad-CAM + SHAP**  
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