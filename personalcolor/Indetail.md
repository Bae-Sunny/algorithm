# 🎨 퍼스널 컬러 분석 알고리즘

## 📋 프로젝트 개요

얼굴 이미지에서 피부색과 눈동자색을 추출하여 개인에게 가장 적합한 퍼스널 컬러 시즌을 AI로 진단하는 프로젝트입니다.

---

## 🔍 프로젝트 목표

### 1️⃣ 정확한 컬러 추출 알고리즘 개발

- 얼굴 영역에서 눈, 눈썹, 입을 제외한 부분의 평균 피부색 추출
- 눈동자 색상 정확히 분석하여 컬러값 추출
- 개인의 컬러 특성을 객관적으로 수치화

### 2️⃣ 퍼스널 컬러 시즌 분류 모델 구현

- 피부색과 눈동자색 기반 4계절 컬러 시스템 분류
- 논문 기반 RGB 표준값을 활용한 과학적 접근
- 시즌별 확률 계산으로 정확도 높은 결과 제공

### 3️⃣ 직관적인 결과 시각화

- 추출된 피부색, 눈동자색 시각화
- 시즌별 확률 그래프로 분석 결과 표현
- 사용자 친화적인 결과 제공

---

## 💻 기술 스택

### 컴퓨터 비전 및 이미지 처리

- **Python**: 주요 개발 언어
- **OpenCV (cv2)**: 이미지 처리 및 얼굴 영역 마스킹
- **dlib**: 얼굴 감지 및 랜드마크 추출
- **NumPy**: 배열 처리 및 계산

### 머신러닝 및 데이터 분석

- **scikit-learn**: 유클리드 거리 계산 및 데이터 분석
- **Matplotlib/Seaborn**: 데이터 시각화
- **pickle**: 모델 저장 및 불러오기

### 개발 환경

- **Google Colab**: 프로토타입 개발 및 테스트
- **Django**: 백엔드 서버 (계획)
- **Vue.js**: 프론트엔드 인터페이스 (계획)
- **Highcharts**: 웹 기반 결과 시각화 (계획)

---

## 🧠 알고리즘 설계 원리

### 📌 퍼스널 컬러의 과학적 접근

- **논문 기반**: "Correlation between the Factors of Personal Color Diagnosis Guide and Brain Wave Analysis"
- **상관관계 분석**: 피부색, 두피색, 눈동자색, 머리카락 색 중 피부색과 눈동자색이 높은 상관관계 보유
- **가중치 설정**: 피부색(70%), 눈동자색(30%)으로 중요도 설정

### 📌 컬러 추출 과정

1. **얼굴 감지**: dlib의 face detector로 얼굴 영역 식별
2. **랜드마크 추출**: 68개 얼굴 랜드마크 포인트 추출
3. **마스킹 처리**: 눈, 눈썹, 입 영역을 제외한 피부 영역 마스크 생성
4. **컬러 추출**: 마스크 영역의 평균 RGB 값 계산
5. **눈동자 영역**: 눈 랜드마크 기반으로 눈동자 영역 특정 및 색상 추출

### 📌 시즌 분류 로직

- **4계절 기준 색상**: 각 시즌별 대표 피부색 및 눈동자색 RGB 값 정의
- **유클리드 거리 계산**: 추출된 색상과 각 시즌 대표색 간의 거리 측정
- **가중 평균**: 피부색과 눈동자색 거리에 가중치 적용
- **확률 계산**: 거리의 역수를 활용한 시즌별 확률 계산

---

## 🔄 구현 과정

### 1️⃣ 얼굴 감지 및 랜드마크 추출

```python
def detect_face_and_landmarks(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 감지
    faces = face_detector(image)
    if len(faces) == 0:
        raise ValueError("얼굴을 감지할 수 없습니다.")

    face = faces[0]
    landmarks = landmark_predictor(image, face)
    return image, face, landmarks

```

### 2️⃣ 피부색 및 눈동자색 추출

```python
def extract_skin_and_eye_colors(image, landmarks):
    # 얼굴 영역 마스크 생성
    face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = []

    # 얼굴 윤곽선 포인트 추출
    for i in range(17):
        points.append((landmarks.part(i).x, landmarks.part(i).y))
    for i in range(26, 16, -1):
        points.append((landmarks.part(i).x, landmarks.part(i).y))

    # 마스크 생성 및 눈, 입 영역 제외
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(face_mask, [points], 255)

    # 눈, 눈썹, 입 영역 제외
    eyes_mouth_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(36, 48): # 눈
        cv2.circle(eyes_mouth_mask, (landmarks.part(i).x, landmarks.part(i).y), 5, 255, -1)
    for i in range(48, 68): # 입
        cv2.circle(eyes_mouth_mask, (landmarks.part(i).x, landmarks.part(i).y), 5, 255, -1)
    for i in range(17, 27): # 눈썹
        cv2.circle(eyes_mouth_mask, (landmarks.part(i).x, landmarks.part(i).y), 5, 255, -1)

    # 최종 마스크 생성 및 색상 추출
    face_mask = cv2.subtract(face_mask, eyes_mouth_mask)
    skin_color = cv2.mean(image, mask=face_mask)[:3]

    # 눈동자색 추출
    left_eye = np.mean(image[landmarks.part(37).y:landmarks.part(41).y,
                      landmarks.part(36).x:landmarks.part(39).x], axis=(0, 1))
    right_eye = np.mean(image[landmarks.part(43).y:landmarks.part(47).y,
                       landmarks.part(42).x:landmarks.part(45).x], axis=(0, 1))
    eye_color = np.mean([left_eye, right_eye], axis=0)

    return skin_color, eye_color, face_mask

```

### 3️⃣ 퍼스널 컬러 시즌 계산

```python
def calculate_season(skin_color, eye_color):
    seasons = {
        "Spring": {
            "skin": [(251, 211, 168), (255, 202, 149), (253, 197, 161), (252, 204, 130)],
            "eye": [(179, 134, 48), (157, 92, 18)]
        },
        "Summer": {
            "skin": [(253, 231, 174), (255, 219, 192), (254, 217, 170), (254, 210, 122)],
            "eye": [(111, 86, 40), (145, 112, 28)]
        },
        "Autumn": {
            "skin": [(255, 221, 150), (247, 206, 152), (249, 201, 128), (212, 169, 101)],
            "eye": [(157, 114, 12), (134, 96, 3)]
        },
        "Winter": {
            "skin": [(255, 220, 147), (242, 206, 148), (247, 207, 121), (216, 173, 102)],
            "eye": [(157, 111, 10), (136, 101, 10)]
        }
    }

    skin_weight, eye_weight = 0.7, 0.3
    distances = {}

    for season, colors in seasons.items():
        skin_distances = [euclidean_distances([skin_color], [s])[0][0] for s in colors["skin"]]
        eye_distances = [euclidean_distances([eye_color], [e])[0][0] for e in colors["eye"]]
        distances[season] = skin_weight * min(skin_distances) + eye_weight * min(eye_distances)

    return min(distances, key=distances.get)

```

### 4️⃣ 시즌별 확률 계산

```python
def calculate_season_probabilities(skin_color, eye_color):
    # 시즌별 거리 계산 (위 코드와 동일)
    # ...

    # 거리의 역수를 사용하여 확률 계산
    total = sum(1/d for d in distances.values())
    probabilities = {season: (1/d)/total * 100 for season, d in distances.items()}

    return probabilities

```

---

## 📊 결과 시각화

### 컬러 분석 과정 시각화
<img width="1361" alt="project6" src="https://github.com/user-attachments/assets/0a6f5462-7c15-4998-a9df-66af1518e824" />

```python
def visualize_results(image, face, skin_color, eye_color, season, face_mask):
    fig, axs = plt.subplots(1, 6, figsize=(30, 5))
    fig.suptitle(f'Personal Color Analysis: {season}', fontsize=16)

    # 원본 이미지
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # 얼굴 감지 이미지
    face_image = image.copy()
    cv2.rectangle(face_image, (face.left(), face.top()),
                  (face.right(), face.bottom()), (0, 255, 0), 2)
    axs[1].imshow(face_image)
    axs[1].set_title('Detected Face')
    axs[1].axis('off')

    # 피부색 추출 마스크
    axs[2].imshow(face_mask, cmap='gray')
    axs[2].set_title('Skin Color Extraction Mask')
    axs[2].axis('off')

    # 추출된 피부색
    skin_patch = np.full((100, 100, 3), skin_color, dtype=np.uint8)
    axs[3].imshow(skin_patch)
    axs[3].set_title(f'Skin Color: RGB{tuple(np.round(skin_color).astype(int))}')
    axs[3].axis('off')

    # 추출된 눈동자색
    eye_patch = np.full((100, 100, 3), eye_color, dtype=np.uint8)
    axs[4].imshow(eye_patch)
    axs[4].set_title(f'Eye Color: RGB{tuple(eye_color.astype(int))}')
    axs[4].axis('off')

    # 시즌 팔레트
    seasons = analyzer.seasons
    palette = np.zeros((100, 100 * len(seasons[season]["skin"]), 3), dtype=np.uint8)
    for i, color in enumerate(seasons[season]["skin"]):
        palette[:, i*100:(i+1)*100] = color
    axs[5].imshow(palette)
    axs[5].set_title(f'{season} Palette')
    axs[5].axis('off')

    plt.show()

```

### 확률 그래프 시각화

<img width="889" alt="project5" src="https://github.com/user-attachments/assets/6eccbaf2-eb6e-4b98-93bf-c29050faa321" />


```python
def visualize_probabilities(probabilities):
    seasons = list(probabilities.keys())
    probs = list(probabilities.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(seasons, probs)
    plt.title('Personal Color Season Probabilities')
    plt.xlabel('Seasons')
    plt.ylabel('Probability (%)')
    plt.ylim(0, 100)

    # 바 위에 확률 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

    plt.show()

```

---

## 🔬 테스트 결과

테스트 이미지별로 다양한 퍼스널 컬러 시즌이 정확하게 진단되었습니다:

<img width="737" alt="스크린샷 2025-02-26 오전 2 32 48" src="https://github.com/user-attachments/assets/2c2f3a7a-a90d-4a4a-aeb2-f3f4cb7e73d4" />


| 테스트 | 진단 결과 | 특징 |
| --- | --- | --- |
| 테스트 1 | Spring | 따뜻하고 밝은 피부톤, 선명한 눈동자색 |
| 테스트 2 | Summer | 차갑고 부드러운 피부톤, 연한 눈동자색 |
| 테스트 3 | Autumn | 따뜻하고 깊이감 있는 피부톤, 짙은 눈동자색 |
| 테스트 4 | Winter | 차갑고 선명한 피부톤, 강한 대비의 눈동자색 |

각 시즌별 분류가 정확하게 이루어졌으며, 확률 분포를 통해 경계선상의 케이스도 확인할 수 있었습니다.

---

## 💡 기술적 도전과 해결 방법

### 1️⃣ 얼굴 랜드마크 정확도 향상

- **문제**: 얼굴 각도나 조명에 따라 랜드마크 추출 정확도 저하
- **해결**: dlib의 shape_predictor_68_face_landmarks 모델 활용 및 전처리 단계 추가
- **결과**: 다양한 조건에서도 안정적인 랜드마크 추출 성공

### 2️⃣ 피부색 추출 정확도 개선

- **문제**: 눈, 입, 눈썹 등 얼굴 요소가 피부색 추출에 영향
- **해결**: 정밀한 마스킹 기법으로 순수 피부 영역만 추출하도록 알고리즘 개선
- **결과**: 보다 정확한 피부색 분석 가능

### 3️⃣ 공통 색상 표준 수립

- **문제**: 퍼스널 컬러 시즌별 대표 색상의 객관적 기준 부재
- **해결**: 논문 데이터 기반 RGB 표준값 설정 및 가중치 체계 도입
- **결과**: 과학적이고 일관된 분류 시스템 구축

---

## 🔍 모델 패키징 및 배포

### 🧪 클래스 기반 모델 구현

```python
class PersonalColorAnalyzer:
    def __init__(self, face_detector, landmark_predictor):
        self.face_detector = face_detector
        self.landmark_predictor = landmark_predictor
        self.seasons = {
            # 시즌별 기준 색상 정의
            # ...
        }
        self.skin_weight = 0.7
        self.eye_weight = 0.3

    def extract_colors(self, image_path):
        # 색상 추출 로직
        # ...

    def calculate_season(self, skin_color, eye_color):
        # 시즌 계산 로직
        # ...

    def calculate_season_probabilities(self, skin_color, eye_color):
        # 확률 계산 로직
        # ...

    def analyze(self, image_path):
        # 통합 분석 메소드
        # ...

```

### 📦 모델 저장 및 배포

```python
# 모델 인스턴스 생성 및 저장
analyzer = PersonalColorAnalyzer(face_detector, landmark_predictor)
with open('personal_color_analyzer.pkl', 'wb') as file:
    pickle.dump(analyzer, file)

# 모델 로드 및 사용
with open('personal_color_analyzer.pkl', 'rb') as file:
    analyzer = pickle.load(file)
season, probabilities, ... = analyzer.analyze(image_path)

```

---

## 📱 웹 서비스 구현 계획

### 백엔드 (Django)

- 이미지 업로드 및 처리 API
- 머신러닝 모델 통합
- 결과 JSON 응답 구조 설계

### 프론트엔드 (Vue.js)

- 사용자 친화적 UI/UX
- 이미지 업로드 컴포넌트
- Highcharts를 활용한 결과 시각화
    
    ![스크린샷 2025-02-26 오전 12.58.34.png](attachment:d1201ad4-fd15-4354-9814-5e908747a42b:스크린샷_2025-02-26_오전_12.58.34.png)
    

---

## 🔮 향후 개선 방향

### 1️⃣ 모델 정확도 향상

- 더 다양한 인종과 피부색에 대한 학습 데이터 확보
- 머리카락 색상 분석 기능 추가
- 조명 보정 알고리즘 개발

### 2️⃣ 기능 확장

- 개인별 어울리는 색상 팔레트 추천
- 메이크업 및 의상 조합 시뮬레이션
- 컬러 칩 기반 색상 매칭 시스템

### 3️⃣ 플랫폼 확장

- 모바일 앱 개발
- 화장품 브랜드 연계 서비스
- API 서비스 제공

---

## 👨‍💻 개발 환경 및 리소스

### 개발 도구

- Google Colab (프로토타이핑)
- PyCharm (백엔드 개발)
- VS Code (프론트엔드 개발)

### 주요 라이브러리 버전

- dlib 19.24.0
- OpenCV 4.8.0
- scikit-learn 1.2.2
- NumPy 1.25.2
- Matplotlib 3.7.1

### 참고 자료

- "Correlation between the Factors of Personal Color Diagnosis Guide and Brain Wave Analysis" (https://e-ajbc.org/journal/view.php?doi=10.20402/ajbc.2016.0071)
- dlib 얼굴 랜드마크 모델: shape_predictor_68_face_landmarks.dat

---

## 📜 Copyright  
© 2024 Bae-Sunny. All rights reserved.
