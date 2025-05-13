## 🎤 **인터뷰 답변 분석기**

### 📌 프로젝트 목표 및 개요

> 면접 답변의 평가 기준이 불분명한 문제를 해결하기 위해
> 
> 
> **KoBERT 모델을 파인튜닝하여 면접 응답을 52개 의도 유형으로 분류**하고
> 
> 직무 연관도, 감정 분석, 키워드 누락까지 분석해 **실용적인 피드백을 생성**했습니다.
> 

---

### 🎯 주요 기능

- **KoBERT 기반 의도/표현/직무 분석 (52개 유형 분류)**
- **형태소 분석 + 감정 사전** 기반 감정 점수 계산
- **직무 키워드 매칭 + 누락 키워드 추천**
- **내용/태도/구체성/언어 사용** 측면의 구조화된 피드백 제공
- **입력 문장 하나로 실시간 분석 결과 도출 가능**

---

### 🧩 기술 스택

- **Core ML**: KoBERT (klue/bert-base) · TensorFlow/Keras
- **Text Processing**: KoNLPy(Okt) · TF-IDF (Scikit-learn)
- **Data Handling**: NumPy · Pandas
- **Resources**: 감정 사전(DB), 직무별 키워드 DB
- **Dev Env**: Google Colab

### ⚙️ 기술 선택 요약

> 한국어 인터뷰 환경에 최적화된 KoBERT를 기반으로
> 
> 
> 다양한 NLP 기법과 **도메인 지식(직무 키워드, 감성 사전)**을 결합하여
> 
> 단순 분석을 넘어 **실제 도움이 되는 피드백 생성 시스템**을 구현했습니다.
> 

---

### 👨‍💻 주요 역할 및 기여

- **면접 답변용 분류 태스크 구성 및 KoBERT 모델 파인튜닝**
- **형태소 분석 기반 감정 점수 계산 로직 설계**
- **직무 키워드 매칭 알고리즘 및 피드백 생성기 구현**
- **모델 학습/검증 및 성능 평가(학습 곡선/혼동 행렬 분석)**
- **사용자 입력 문장 하나로 종합 분석 결과 도출되는 파이프라인 구현**

---

### 🏆 주요 성과

- **KoBERT 기반 의도 분류 정확도: 훈련 96.4%, 검증 74.1%**
- **자체 감정 사전 기반 분석으로 피드백 다양성 확보**
- **직무별 피드백 제안 기능으로 실용성 강화**
- **NLP 모델을 실제 응답 분석에 적용한 실전 사례 구축**

---

### 💡 기술적 도전 및 해결

- **의도 분류 성능 확보 (다중 클래스, 52개)**
    
    → [CLS] + GlobalAvgPooling 결합 및 잔차 연결 구조로 정확도 확보
    
- **감정 분석 로직의 정밀도 개선**
    
    → 감정 사전 + 긍정 문구 패턴 인식으로 정확도 보완
    
- **직무 키워드 누락 제안 자동화**
    
    → 유의미 단어 추출 → 매칭 실패 키워드 기반 개선 제안 구성
    

---

### 📈 모델 구조 및 처리 흐름

1. **KoBERT 기반 답변 분석 모델 아키텍처**

- **파인튜닝:** 한국어 면접 답변 데이터셋을 사용하여 사전 학습된 KoBERT 모델(`klue/bert-base`)을 미세 조정했습니다.
- **입력 처리:** 답변 텍스트를 최대 128 토큰 길이로 처리하며, 패딩 및 트런케이션을 적용했습니다.
- **모델 구조:**
    - BERT 모델의 [CLS] 토큰 출력(Pooled Output)과 모든 토큰의 평균값(Global Average Pooling)을 결합하여 문맥 정보를 풍부하게 활용했습니다.
    - 결합된 특징 벡터는 잔차 연결(Residual Connection)과 GELU 활성화 함수가 적용된 여러 Dense 레이어를 통과하며 정제됩니다. (Dropout으로 과적합 방지)
    - 최종적으로 52개의 답변 의도 클래스를 분류하기 위해 Softmax 활성화 함수를 사용했습니다.

2. **분석 처리 흐름**

```
[면접 답변 입력] → [텍스트 전처리 (형태소 분석, 불용어 제거)] → [KoBERT 모델 예측 (의도/표현/직무)] → [키워드 추출 (TF-IDF, 직무 키워드 매칭)] → [감정 분석 (감정 사전)] → [종합 피드백 생성]
```

---

### **🔄 구현 과정 하이라이트**

1. **데이터 준비 및 리소스 로드:** 면접 답변 의도 레이블(`full_intent_labels.tsv`), KoBERT 토크나이저 및 파인튜닝된 모델, 직무별 키워드 DB, 감정 사전 등을 로드하여 분석 환경을 구성했습니다.
    
    ```python
    # 주요 라이브러리 및 모델 로드
        from transformers import BertTokenizer, TFBertModel
        from tensorflow.keras.models import load_model
        from konlpy.tag import Okt
        import numpy as np
        import pandas as pd
    
        tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
        # best_intent_model.h5 파일 로드 (경로 확인 필요)
        # model = load_model('/path/to/best_intent_model.h5', custom_objects={'TFBertModel': TFBertModel}) 
        okt = Okt()
        # 기타 데이터 로드 (레이블, 키워드, 감정 사전 등)
    ```
1. **핵심 분석 함수 구현**
    - `perdict(sentence)`: 입력된 답변 문장을 받아 KoBERT 모델로 분석하여 카테고리, 표현, 직무, 설명을 예측합니다.
        
        ```python
        def predict(sentence):
            inputs = tokenizer(
                sentence,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='tf'
            )
            out = model({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']})
            logits = out.numpy()
            predicted_class = np.argmax(logits, axis=1)[0]
            prediction_info = full_intent_labels.iloc[predicted_class]
        
            return prediction_info['category'], prediction_info['expression'], prediction_info['description'], prediction_info['job']
        
        ```
        
    - `improved_sentiment_analysis(text)`: 형태소 분석 및 자체 구축한 한국어 감정 사전을 기반으로 답변의 감성 점수를 계산합니다. 특정 긍정 패턴 등장 시 가중치를 부여합니다
        
        ```python
        def improved_sentiment_analysis(text):
            words = okt.morphs(text)
            score = sum(korean_sentiment_dict.get(word, 0) for word in words)
        
            positive_patterns = ['을 높였습니다', '에 기여했습니다', '을 달성했습니다']
            for pattern in positive_patterns:
                if pattern in text:
                    score += 2
        
            return score
        
        ```
        
    - `analyze_response(sentence)`: `perdict`, `improved_sentiment_analysis` 함수 등을 호출하고, 직무 키워드 매칭 및 의미있는 단어 추출을 수행하여 종합적인 분석 결과를 딕셔너리 형태로 반환합니다.
        
        ```python
        def analyze_response(sentence):
            category, expression, description, job = predict(sentence)
        
            # 형태소 분석 및 불용어 제거
            morphs = okt.morphs(sentence)
            meaningful_words = [word for word in morphs if word not in stop_words]
        
            # 직무별 키워드 매칭
            matched_keywords = [keyword for keyword in job_keywords.get(job, [])
                               if keyword in meaningful_words]
        
            # 감정 분석
            sentiment = improved_sentiment_analysis(' '.join(meaningful_words))
        
            return {
                'category': category,
                'expression': expression,
                'job': job,
                'description': description,
                'matched_keywords': matched_keywords,
                'sentiment': sentiment,
                'meaningful_words': meaningful_words
            }
        
        ```
        
    - `generate_feedback(anaysis_result)`: `analyze_response`결과를 바탕으로 내용, 태도, 구체성, 언어 사용의 4가지 측면에서 사용자 친화적인 피드백 텍스트를 생성합니다. 누락된 직무 키워드를 제안하는 로직도 포함됩니다.
        
        ```python
        def generate_feedback(analysis_result):
            category = analysis_result['category']
            expression = analysis_result['expression']
            job = "전체" if analysis_result['job'] == "전 직무" else analysis_result['job']
            description = analysis_result['description']
        
            feedback = f"답변 카테고리: {category_descriptions.get(category, category)}\\n"
            feedback += f"관련 역량/특성: {description} (표현: {expression})\\n"
            feedback += f"관련 직무: {job}\\n\\n"
        
            # 1. 내용 관련 피드백
            feedback += "1. 내용 분석:\\n"
            for keyword in analysis_result['matched_keywords']:
                feedback += f"- {keyword}와(과) 관련된 내용을 언급하여 {job} 직무에 대한 이해를 보여주었습니다.\\n"
        
            # 2. 태도 관련 피드백
            feedback += "\\n2. 태도 분석:\\n"
            if analysis_result['sentiment'] > 3:
                feedback += "- 매우 긍정적이고 열정적인 태도가 잘 드러납니다.\\n"
            elif analysis_result['sentiment'] > 0:
                feedback += "- 긍정적인 태도가 느껴집니다.\\n"
        
            # 3. 구체성 관련 피드백
            feedback += "\\n3. 구체성 분석:\\n"
            if len(analysis_result['meaningful_words']) > 30:
                feedback += "- 답변이 상당히 구체적이고 상세합니다. 좋은 인상을 줄 수 있습니다.\\n"
        
            # 4. 언어 사용 관련 피드백
            feedback += "\\n4. 언어 사용 분석:\\n"
            unique_words = set(analysis_result['meaningful_words'])
            if len(unique_words) > 30:
                feedback += "- 다양한 어휘를 사용하여 풍부한 답변을 제시했습니다.\\n"
        
            return feedback
        ```
        

---

### 📊 모델 성능 요약
![image](https://github.com/user-attachments/assets/6a2c6630-12c0-4ada-9657-85a33080a21b)

- **훈련 정확도:** 96.4%
- **검증 정확도:** 74.1%
- **Validation Loss:** 1.29
- **혼동 행렬 분석 결과:** 일부 유사 카테고리 간 혼동 존재 (예: 성격/배경 관련)

> → 향후 데이터 확대 및 정규화로 과적합 완화 가능
> 

---

### 🧪 사용 예시

Q: 최근 직장에서 가장 힘들었던 일은 무엇이었나요?

A: "최근 직장에서 겪었던 가장 큰 어려움은 팀 프로젝트의 마감 기한이 촉박할 때 발생했습니다. 프로젝트 초기 단계에서 예상보다 많은 문제가 발생했지만, 팀원들과 협력하여 해결했습니다."

분석 결과:

```
답변 카테고리: 태도 관련 역량

관련 역량/특성: 책임감 (표현: f_resp)

관련 직무: 제조/생산

1. 내용 분석:
  - 경험, 스트레스, 협력 키워드를 언급하여 관련 역량을 보여주었습니다.
  - [제조/생산] 직무 관련성을 높이려면 '교대근무', '책임감', '안전' 등의 키워드 활용을 고려해보세요.
2. 태도 분석:
  - 긍정적인 태도가 느껴지는 답변입니다.
3. 구체성 분석:
  - 답변 내용이 구체적이고 상세하여 좋은 인상을 줄 수 있습니다.
4. 언어 사용 분석:
  - 다양한 어휘를 사용하여 답변의 풍부함을 더했습니다.
```

---

### 🌱 성장 및 배움

- **도메인 특화 NLP 설계 능력 강화 (면접 전용 피드백 모델 구축)**
- **텍스트 처리 + 분류 + 피드백 구성 전체 파이프라인 경험**
- **모델 성능 평가 및 과적합 대응 전략 수립 역량 확보**

---

### 🚀 향후 개선 방향

- 고품질 면접 데이터 추가 확보 및 모델 정제
- LLM 기반 모델(BERT-Large, KoAlpaca 등)으로 성능 향상
- **음성 입력 및 실시간 응답 피드백** 기능 추가
- 모바일 앱 형태의 서비스 확장 고려

---

### 🔗 참고 데이터셋 및 자료

- **KoBERT**: klue/bert-base (HuggingFace)
- **형태소 분석**: KoNLPy (Okt)
- **한국어 감정 사전**: 국립국어원, 자체 구축 확장
- **직무 기술서 (NCS 기반 키워드)**
- **논문**: BERT: Pre-training of Deep Bidirectional Transformers

---
