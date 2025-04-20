# 🎤 **인터뷰 답변 분석기**

## 📌 프로젝트 목표 및 개요

면접 준비 시 자신의 답변이 어떻게 평가될지 객관적으로 파악하기 어렵다는 문제점에 착안하여, **AI 자연어 처리 기술**을 활용해 면접 답변의 **의도를 정밀하게 분석**하고 **맞춤형 피드백**을 제공하는 시스템을 개발했습니다. 한국어에 특화된 **KoBERT 모델**을 기반으로, 답변의 카테고리, 사용된 표현, 연관 직무 등을 예측하고 구체적인 개선점을 제시하여 사용자의 성공적인 면접 준비를 돕는 것을 목표로 합니다.

---

## ✨ 주요 기능 및 가치

- **답변의 숨은 의도까지 파악 (KoBERT 기반 분석):** 단순 키워드 매칭을 넘어, KoBERT 모델을 파인튜닝하여 면접 답변에 담긴 **핵심 의도(카테고리), 표현 방식, 연관 직무**를 52가지 유형으로 정밀하게 분류하고 예측합니다.
- **다각적이고 심층적인 피드백 제공:** 답변 내용을 **내용, 태도, 구체성, 언어 사용**의 4가지 핵심 기준으로 분석합니다. 자체 구축한 한국어 감정 사전을 통해 답변의 긍정/부정 뉘앙스를 평가하고, TF-IDF 기반으로 핵심 단어를 추출하여 피드백의 깊이를 더했습니다.
- **실질적인 개선 방향 제시:** 분석 결과를 바탕으로 **직무 관련 누락 키워드를 제안**하고, 답변의 구체성과 표현 방식을 개선할 수 있는 **실용적인 조언**을 제공하여 사용자가 면접에서 더 좋은 성과를 낼 수 있도록 지원합니다.

---

## 🛠️ 기술 스택

- **Core ML/NLP:** Python, TensorFlow/Keras (v2.15.0), Transformers (v4.35.0), KoBERT (klue/bert-base)
- **Text Processing:** KoNLPy (Okt), Scikit-learn (TF-IDF)
- **Data Handling:** NumPy, Pandas
- **Knowledge Base:** 자체 구축 한국어 감정 사전, 직무별 키워드 DB
- **Development Environment:** Google Colab

---

## 📊 모델 구조 및 분석 파이프라인

### 1. **KoBERT 기반 답변 분석 모델 아키텍처**

- **파인튜닝:** 한국어 면접 답변 데이터셋을 사용하여 사전 학습된 KoBERT 모델(`klue/bert-base`)을 미세 조정했습니다.
- **입력 처리:** 답변 텍스트를 최대 128 토큰 길이로 처리하며, 패딩 및 트런케이션을 적용했습니다.
- **모델 구조:**
    - BERT 모델의 [CLS] 토큰 출력(Pooled Output)과 모든 토큰의 평균값(Global Average Pooling)을 결합하여 문맥 정보를 풍부하게 활용했습니다.
    - 결합된 특징 벡터는 잔차 연결(Residual Connection)과 GELU 활성화 함수가 적용된 여러 Dense 레이어를 통과하며 정제됩니다. (Dropout으로 과적합 방지)
    - 최종적으로 52개의 답변 의도 클래스를 분류하기 위해 Softmax 활성화 함수를 사용했습니다.

### 2. **분석 처리 흐름**

```
[면접 답변 입력] → [텍스트 전처리 (형태소 분석, 불용어 제거)] → [KoBERT 모델 예측 (의도/표현/직무)] → [키워드 추출 (TF-IDF, 직무 키워드 매칭)] → [감정 분석 (감정 사전)] → [종합 피드백 생성]
```

---

## 🔄 구현 과정 하이라이트

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
    

![image](https://github.com/user-attachments/assets/bed504f5-81d4-4db4-ad62-622146edcfcb)


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

## 📈 모델 성능 및 평가

- **학습 결과:** 훈련 데이터셋 기준 약 96.4%의 정확도를 달성했으며, 검증 데이터셋 기준 약 74.1%의 정확도를 기록했습니다. 검증 손실(Validation Loss)은 1.29로 나타났습니다.
- **학습 곡선 분석:** 학습이 진행됨에 따라 훈련 손실은 꾸준히 감소하고 훈련 정확도는 증가했으나, 검증 손실은 특정 시점 이후 소폭 상승하는 경향을 보여 추가적인 데이터 확보 또는 정규화 기법 강화를 통한 과적합 방지 노력이 필요함을 시사합니다.
    
   ![image](https://github.com/user-attachments/assets/565fd078-d638-4a65-a1d5-2941571f1016)

    
- **혼동 행렬 분석:** 대부분의 의도 클래스를 잘 분류했지만, 의미적으로 유사한 일부 카테고리(예: 성격 관련과 배경 관련 답변) 사이에서는 다소 혼동이 발생하는 것을 확인했습니다.

---

## 💬 사용 예시 (피드백 생성 결과)

**입력:** "최근 직장에서 겪었던 가장 큰 어려움은 팀 프로젝트의 마감 기한이 촉박할 때 발생했습니다. 프로젝트 초기 단계에서 예상보다 많은 문제가 발생했지만, 팀원들과 협력하여 해결했습니다."

**출력 피드백:**

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

## 🌟 프로젝트 특징 및 차별점

- **다차원적 분석:** 단순 키워드나 감성 분석을 넘어, 답변의 **구조적 의도, 표현 방식, 직무 연관성**까지 종합적으로 분석하여 피드백의 깊이를 더했습니다.
- **직무 맞춤형 피드백:** 7개 주요 직무 분야의 핵심 키워드 DB를 활용하여 **지원하는 직무에 맞춰** 강점을 부각하고 약점을 보완할 수 있도록 실질적인 조언을 제공합니다.
- **한국어 특화 모델:** **KoBERT**와 **KoNLPy(형태소 분석기)**, 자체 구축 **한국어 감정 사전**을 활용하여 한국어 면접 환경에 최적화된 분석 성능을 제공합니다.

---

## 🌱 성장 및 배움 (Key Takeaways)

- **최신 NLP 모델 활용 능력:**KoBERT와 같은 사전 학습된 언어 모델을 특정 도메인(면접 답변)에 맞게 파인튜닝하고 적용하는 경험을 통해 최신 NLP 기술 활용 역량을 강화했습니다.
- **도메인 특화 데이터 구축:** 면접이라는 특정 상황에 맞는 데이터(레이블, 키워드 DB, 감정 사전)를 정의하고 구축하는 과정을 통해 도메인 지식의 중요성을 깨달았습니다.
- **모델 성능 분석 및 개선:** 학습 곡선, 혼동 행렬 등 다양한 지표를 통해 모델의 성능을 객관적으로 평가하고, 문제점을 진단하여 개선 방향을 도출하는 능력을 길렀습니다.
- **AI 기반 피드백 시스템 설계:** 분석된 결과를 바탕으로 사용자에게 실질적인 도움을 줄 수 있는 피드백 로직을 설계하고 구현하는 경험을 했습니다.

### 🚀 향후 개선 방향

1. **모델 성능 향상:** 더 많은 고품질 면접 데이터를 확보하여 모델을 재학습시키고, 하이퍼파라미터 튜닝, 최신 LLM(대규모 언어 모델) 적용 등을 통해 분석 정확도를 높입니다.
2. **실시간 상호작용 기능:** 음성 인식 기술을 통합하여 사용자가 말하는 즉시 피드백을 제공하고, 비디오 분석을 통해 자세, 시선 처리 등 비언어적 요소까지 평가하는 기능을 추가합니다.
3. **서비스 확장:** 웹 또는 모바일 애플리케이션으로 개발하여 사용자의 접근성을 높이고, 개인별 면접 준비 진행 상황을 관리하고 맞춤형 학습 계획을 추천하는 기능을 구현합니다.

---

## 🔗 관련 자료 및 링크

**참고 논문/자료:**
*   BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
*   KoBERT 관련 기술 문서
*   한국어 감성 사전 관련 자료 (국립국어원 등)
*   NCS 직무 기술서 (직무 키워드 참고)

---
