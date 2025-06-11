## ☕ **커피 원두 등급 예측 서비스**
![Beige Yellow Elegant Portofolio](https://github.com/user-attachments/assets/3189aa2b-0742-47b6-9d5b-f26abe4d2379)

### 📌 프로젝트 목표 및 개요

> 커피 품질 평가의 주관성을 줄이기 위해
> 
> 
> **객관적인 특성 데이터 기반의 등급 예측 모델**(KNN)을 학습하고
> 
> **사용자 입력을 바탕으로 품질을 예측하는 웹 서비스**를 개발했습니다.
> 

---

### 🎯 주요 기능

- **9가지 특성 기반 원두 등급 예측 (KNN)**
    
    → 향, 맛, 산미, 바디감 등 커피 특성을 입력하면 **스페셜티/프리미엄/상업용 등급 예측**
    
- **직관적인 웹 입력 인터페이스 (Vue.js)**
    
    → 필드 유효성 검증 포함된 **9개 항목 입력 폼**과 예측 결과 시각화 제공
    
- **모델 기반 API 통합 (Django)**
    
    → 학습된 KNN 모델을 백엔드에 연동하여 **RESTful API로 예측 결과 반환**
    
- **사용자 경험 강화 기능**
    
    → 예측 등급별 설명, 결과 이미지, 향후 추천 기능까지 고려한 확장 구조 설계
    

---

### 🧩 기술 스택

- **Frontend**: Vue.js · Axios · HTML5 · CSS3
- **Backend**: Django · Joblib · REST API
- **AI / ML**: Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn
- **Dev Tools**: Google Colab · VS Code · PyCharm
- **Collaboration**: Git · GitHub

### ⚙️ 기술 선택 요약

> KNN 알고리즘은 유사도 기반 분류에 강점을 보여 커피 등급 분류에 적합했고,
> 
> 
> **Scikit-learn + Joblib**으로 모델 학습 및 배포 구조를 구성했습니다.
> 
> **Django + Vue.js**로 프론트/백의 완전한 풀스택 통합 구조를 구현했습니다.
> 

---

### 👨‍💻 주요 역할 및 기여

- **커피 데이터셋 전처리 및 이상치 제거 (EDA)**
- **KNN 모델 학습 및 다양한 K값 실험, 최적 K=57 선정**
- **Django API 구현 및 CORS 대응 설정**
- **Vue.js 기반 입력 폼 및 결과 화면 개발**
- **모델 성능 비교 및 결과 문서화**

---

### 🏆 주요 성과

- **KNN 알고리즘 정확도 87.84% 달성 (K=57 기준)**
- **최적 K값 도출을 통한 성능 고도화 및 과적합 방지**
- **Vue–Django 기반 풀스택 예측 서비스 완성**
- **모델과 사용자 간 직접 상호작용 가능한 AI 서비스 제공**

---

### 💡 기술적 도전 및 해결

- **K값 최적화로 모델 성능 극대화**
    
    → 1~103 구간 실험을 통해 최적 K=57 선정
    
- **데이터 품질 확보**
    
    → IQR 기반 이상치 제거 및 결측치 보완으로 정확도 향상
    
- **프론트–백 통신 문제 해결**
    
    → Django의 CORS 설정을 통해 **API 요청 오류 해결**
    

---

### 📊 모델 성능 비교

| 모델 | 정확도 | 특징 |
| --- | --- | --- |
| **KNN (K=57)** | **87.84%** | ✅ 최종 선정 모델 |
| Logistic Regression | 86.04% | max_iter=1000 |
| Decision Tree | 81.98% | max_depth=5 |
| Voting Classifier | 86.76% | Soft Voting 기반 앙상블 |

> KNN이 커피 품질과 특성 간의 거리 기반 유사성을 가장 잘 반영함
> 

---

### 📁 활용 데이터셋

- **Kaggle - Coffee Quality Database**
    
    → 아라비카·로부스타 원두의 품질 특성과 Total Cup Points 포함
    
    → [🔗 Kaggle 링크](https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi/data)
    

---

### 🌱 성장 및 배움

- **모델 개발부터 배포까지 MLOps 전체 주기 경험**
- **데이터 기반 의사결정 및 모델 비교·선택 능력 향상**
- **풀스택 프로젝트의 구조 이해 및 문제 해결 역량 강화**

---

### 🚀 향후 개선 방향

- 딥러닝 기반 TabNet 등 고성능 모델 도입
- 사용자 맞춤 커피 추천 기능 및 과거 이력 기반 진단 강화
- 피드백 기반 모델 보정 기능 추가

---
