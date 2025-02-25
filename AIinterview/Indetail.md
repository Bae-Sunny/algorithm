# 🎤 인터뷰 답변 분석 및 피드백 시스템
## 📋 프로젝트 개요

KoBERT 모델을 활용하여 면접 답변의 의도를 자동으로 분석하고 맞춤형 피드백을 제공하는 AI 시스템입니다. 면접자의 답변을 자연어 처리 기술로 분석하여 카테고리, 표현, 관련 직무를 예측하고 구체적인 개선 방향을 제시합니다.

---

## 🔍 프로젝트 목표

### 1️⃣ 정확한 면접 답변 분석

- KoBERT 기반 맞춤형 모델로 면접 답변의 숨은 의도 파악
- 카테고리, 표현 방식, 관련 직무를 자동으로 분류
- 직무별 키워드 매칭 및 활용도 측정

### 2️⃣ 다각적 피드백 제공

- 내용, 태도, 구체성, 언어 사용 등 다양한 측면에서 분석
- 감정 분석을 통한 긍정/부정 표현 평가
- 직무 관련성 및 언급된 키워드 분석

### 3️⃣ 실용적인 개선 방향 제시

- 누락된 직무 관련 키워드 제안
- 답변의 구체성 및 표현 방식 개선 방향 제시
- 종합적인 면접 피드백 생성

---

## 💻 기술 스택

### 자연어 처리 및 딥러닝

- **Python**: 주요 개발 언어
- **TensorFlow/Keras (v2.15.0)**: 딥러닝 모델 구현
- **Transformers (v4.35.0)**: BERT 모델 및 토크나이저 활용
- **KoBERT**: 한국어 텍스트 처리를 위한 사전학습 모델

### 텍스트 처리 및 분석

- **KoNLPy**: 한국어 형태소 분석
- **scikit-learn**: TF-IDF 벡터화 및 기타 머신러닝 기능
- **NumPy/Pandas**: 데이터 처리 및 분석

### 감정 분석 및 데이터베이스

- **한국어 감정 사전**: 자체 구축한 감정 어휘 사전
- **직무별 키워드 데이터베이스**: 각 직무 분야별 핵심 키워드 집합

---

## 📊 모델 구조 및 알고리즘

### 1️⃣ BERT 기반 의도 분석 모델

- **KoBERT 파인튜닝**: 면접 답변 데이터로 사전학습된 BERT 모델 미세조정
- **입력 처리**: 최대 128 토큰, 패딩 및 트런케이션 적용
- **아키텍처**:
    - BERT의 풀링된 출력과 글로벌 평균 풀링 출력을 결합
    - 잔차 연결(Residual Connection)이 포함된 Dense 레이어 층
    - GELU 활성화 함수와 Dropout 정규화
    - 52개 클래스 분류를 위한 소프트맥스 출력

### 2️⃣ 분석 파이프라인

```
입력 답변 → 전처리 → BERT 모델 → 의도 분류 → 키워드 추출 → 감정 분석 → 피드백 생성

```

### 3️⃣ 핵심 알고리즘

- **의도 분류**: 답변의 카테고리(attitude, background, personality 등)와 표현(f_cons, c_person 등) 예측
- **키워드 매칭**: 직무별 핵심 키워드와 답변 내용의 일치도 분석
- **감정 점수 계산**: 한국어 감정 사전과 패턴 매칭을 통한 감정 극성 평가
- **TF-IDF 기반 중요 단어 추출**: 답변에서 가장 중요한 단어들을 식별

---

## 🔄 구현 과정

### 1️⃣ 데이터 준비 및 모델 로드

- 레이블 데이터(`full_intent_labels.tsv`) 로드: 카테고리, 표현, 설명, 직무 정보 포함
- BERT 토크나이저와 사전 훈련된 모델 로드
- 직무별 키워드, 카테고리 설명, 감정 사전 등 정의

```python
# BERT 토크나이저 및 모델 로드
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
with tf.keras.utils.custom_object_scope({'TFBertModel': TFBertModel}):
    model = load_model('/path/to/best_intent_model.h5')

# 한국어 형태소 분석기 초기화
okt = Okt()

```
<img width="712" alt="스크린샷 2025-02-26 오전 2 37 28" src="https://github.com/user-attachments/assets/97526995-004d-495c-914a-1d0026f4bbfc" />


### 2️⃣ 주요 함수 구현

### 의도 예측 함수

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

### 감정 분석 함수

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

### 종합 분석 함수

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

### 3️⃣ 피드백 생성

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

## 📈 모델 성능 평가

### 모델 학습 결과

- **훈련 정확도**: 96.42%
- **검증 정확도**: 74.05%
- **훈련 손실**: 0.0518
- **검증 손실**: 1.2918

### 학습 곡선

<img width="889" alt="project5" src="https://github.com/user-attachments/assets/34318bb3-dc21-49a6-b9e6-5ef67daea149" />


### 시즌별 혼동 행렬

- 대부분의 클래스에서 높은 정확도 달성
- 일부 유사한 카테고리 간 혼동 발생 (특히 'personality'와 'background' 사이)

---

## 📱 사용 예시

### 입력 예시 1

> "최근 직장에서 겪었던 가장 큰 어려움은 팀 프로젝트의 마감 기한이 촉박할 때 발생했습니다. 프로젝트 초기 단계에서 예상보다 많은 문제가 발생했지만, 팀원들과 협력하여 해결했습니다."
> 

### 출력 피드백

```
답변 카테고리: 태도 관련 역량
관련 역량/특성: 책임감 (표현: f_resp)
관련 직무: 제조/생산

1. 내용 분석:
- 다음 키워드들도 언급되어 다양한 역량을 보여주었습니다: 경험, 스트레스, 협력
- 제조/생산 직무와 더 관련성을 높이기 위해 다음 키워드들을 고려해보세요: 교대근무, 책임감, 안전

2. 태도 분석:
- 긍정적인 태도가 느껴집니다.

3. 구체성 분석:
- 답변이 상당히 구체적이고 상세합니다. 좋은 인상을 줄 수 있습니다.

4. 언어 사용 분석:
- 다양한 어휘를 사용하여 풍부한 답변을 제시했습니다. 이는 해당 분야에 대한 깊은 이해를 보여줍니다.

```

### 입력 예시 2

> "제가 겪었던 실패는 신규 고객 관리 시스템 도입 프로젝트에서 발생했습니다. 사용자 요구사항을 충분히 반영하지 않아 시스템이 실제 업무에 맞지 않았습니다."
> 

### 출력 피드백

```
답변 카테고리: 지식/기술 관련 역량
관련 역량/특성: 장애 대응 (표현: i_dis_coping)
관련 직무: ICT

1. 내용 분석:
- 시스템와(과) 관련된 내용을 언급하여 ICT 직무에 대한 이해를 보여주었습니다.
- 다음 키워드들도 언급되어 다양한 역량을 보여주었습니다: 관리, 고객, 경험
- ICT 직무와 더 관련성을 높이기 위해 다음 키워드들을 고려해보세요: 기술, 프로그래밍, 알고리즘

2. 태도 분석:
- 긍정적인 태도가 느껴집니다.

3. 구체성 분석:
- 답변이 상당히 구체적이고 상세합니다. 좋은 인상을 줄 수 있습니다.

4. 언어 사용 분석:
- 다양한 어휘를 사용하여 풍부한 답변을 제시했습니다. 이는 해당 분야에 대한 깊은 이해를 보여줍니다.

```

---

## 🌟 프로젝트 특징 및 차별점

### 1️⃣ 다차원적 분석 접근

- 단순 감정 분석을 넘어 의도, 직무 관련성, 표현 방식까지 종합 분석
- 4가지 핵심 영역(내용, 태도, 구체성, 언어 사용)에서 균형 잡힌 피드백 제공

### 2️⃣ 직무 맞춤형 피드백

- 7개 주요 직무 분야별 키워드 데이터베이스 활용
- 직무별 필요 역량과 표현 방식에 대한 맞춤형 조언
- 누락된 직무 관련 키워드 제안을 통한 실용적 가이드 제공

### 3️⃣ 한국어 특화 구현

- KoBERT를 활용한 한국어 면접 답변 특화 모델
- 한국어 형태소 분석 기반의 정확한 키워드 추출
- 한국어 감정 사전을 활용한 섬세한 감정 분석

---

## 🔍 향후 개선 방향

### 1️⃣ 모델 고도화

- 더 많은 면접 데이터로 모델 재학습
- 다양한 산업군 및 직무별 특화 모델 개발
- 더 높은 성능의 대용량 언어 모델(LLM) 통합

### 2️⃣ 기능 확장

- 음성 인식 통합으로 실시간 면접 피드백
- 비디오 분석을 통한 비언어적 요소 평가
- 개인화된 학습 계획 및 면접 준비 가이드 제공

### 3️⃣ 사용성 개선

- 웹 및 모바일 애플리케이션 개발
- 사용자 친화적 인터페이스 구축
- 맞춤형 면접 시뮬레이션 기능 추가

---

## 📚 참고 자료 및 리소스

### 논문 및 기술 문서

- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
- "Correlation between the Factors of Personal Color Diagnosis Guide and Brain Wave Analysis"
- "How to Fine-Tune BERT for Text Classification?" (Sun et al., 2019)

### 데이터 소스

- 한국직업능력개발원의 NCS(국가직무능력표준) 자료
- LinkedIn의 "The Skills Companies Need Most" 연간 보고서
- 국립국어원의 "한국어 감성 사전" 프로젝트
- KOSAC(Korean Sentiment Analysis Corpus) 프로젝트

---

## 📜 Copyright  
© 2024 Bae-Sunny. All rights reserved.
