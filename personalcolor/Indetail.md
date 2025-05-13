## 🎨 퍼스널 컬러 진단 알고리즘

### 📌 프로젝트 목표 및 개요

> AI 기반 얼굴 이미지 분석 알고리즘을 통해
> 
> 
> 전문가 없이도 개인의 **퍼스널 컬러 시즌(봄/여름/가을/겨울)**을
> 
> 과학적 기준에 따라 자동 진단하는 알고리즘을 개발했습니다.
> 

---

### 🎯 주요 기능

- **AI 기반 얼굴 컬러 분석**
    - 얼굴 이미지에서 **피부색(70%)과 눈동자색(30%)**을 추출
    - 표준 RGB값과의 **유클리드 거리**를 계산하여 컬러 시즌 분류
- **시각적 피드백 제공**
    - 추출된 색상, 컬러 시즌 확률, 추천 팔레트 등을 **그래프 및 이미지로 시각화**
- **알고리즘 클래스화 및 재사용 가능 구조 설계**
    - 진단 로직을 Python 클래스(`PersonalColorAnalyzer`)로 구성하여 **모델 저장 및 배포에 최적화**

---

### 🧩 기술 스택

- **AI / CV**: OpenCV · dlib · scikit-learn
- **Data Analysis**: NumPy · Matplotlib · Pickle
- **Dev Env**: Google Colab
- **(Web 확장 계획)**: Django · Vue.js · Highcharts

---

### ⚙️ 알고리즘 설계 요약

> dlib 랜드마크 기반 마스킹으로 피부·눈동자 RGB 평균 추출
> 
> 
> → 표준 컬러 프로파일과 유클리드 거리 계산
> 
> → **가중치 기반 퍼스널 컬러 시즌 진단**
> 

---

### 🔄 구현 과정 하이라이트

1. **얼굴 및 랜드마크 감지 (`detect_face_and_landmarks`):** dlib 라이브러리를 활용하여 이미지 내 얼굴 영역과 68개의 세부 랜드마크 좌표를 정확히 추출했습니다. 얼굴 미감지 시 예외 처리를 포함했습니다.
    
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
    
2. **정밀한 컬러 추출 (`extract_skin_and_eye_colors`):** 얼굴 윤곽선과 눈/입/눈썹 랜드마크를 조합하여 피부 영역 마스크를 생성하고, `cv2.mean` 함수를 마스크와 함께 사용하여 정확한 평균 피부색 및 눈동자색 RGB 값을 계산했습니다.
    
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
    
3. **시즌 분류 로직 구현 (`calculate_season`):** 4계절별 표준 색상값 딕셔너리를 정의하고, `sklearn.metrics.pairwise.euclidean_distances`를 사용하여 추출된 색상과 표준값 간의 거리를 효율적으로 계산했습니다. 피부/눈 가중치를 적용하여 최종 시즌을 결정했습니다.
    
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
    
4. **확률 계산 로직 구현 (`calculate_season_probabilities`):** 시즌별 거리의 역수를 활용하여 확률 분포를 계산하는 함수를 구현했습니다.
    
    ```python
    def calculate_season_probabilities(skin_color, eye_color):
        # 시즌별 거리 계산 (위 코드와 동일)
        # ...
    
        # 거리의 역수를 사용하여 확률 계산
        total = sum(1/d for d in distances.values())
        probabilities = {season: (1/d)/total * 100 for season, d in distances.items()}
    
        return probabilities
    
    ```
    
5. **결과 시각화 (`visualize_results`, `visualize_probabilities`):** Matplotlib을 사용하여 원본 이미지, 얼굴 감지 결과, 피부 마스크, 추출된 색상 패치, 진단된 시즌의 컬러 팔레트, 그리고 시즌별 확률 막대그래프를 생성하여 분석 과정을 명확하게 보여주도록 구현했습니다.
    
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
![image](https://github.com/user-attachments/assets/519e2a71-6bc6-47d1-8fe4-306ff28c29b4)

---

### 📈 테스트 결과

| 이미지 | 특징 | 진단 결과 |
| --- | --- | --- |
| 인물 1 | 밝은 피부, 선명한 눈동자 | **Spring** |
| 인물 2 | 차가운 톤, 어두운 눈동자 | **Summer** |
| 인물 3 | 따뜻한 피부, 갈색 눈동자 | **Autumn** |
| 인물 4 | 대비 강한 얼굴 특징 | **Winter** |

> 분석 결과는 대부분 실제 진단과 일치하며
> 
> 
> **확률 분포 시각화**를 통해 신뢰도 확인 가능
> 
![image](https://github.com/user-attachments/assets/edafad61-77ef-4b52-803b-b0076886460d)

---

### 💡 기술적 도전 및 해결

- **정확한 얼굴 컬러 추출**
    - 눈/입/눈썹 제외 마스킹으로 **순수 피부영역 추출 정밀도 향상**
- **조명/각도 변화 대응**
    - **전처리 + 고성능 랜드마크 모델**로 다양한 조건에서 안정성 확보
- **객관적인 기준 부족 문제**
    - 논문 기반 컬러 기준값 정의 + **유클리드 거리 기반 계량화**로 해결

---

### 🌱 성장 및 배움

- **컴퓨터 비전 알고리즘 개발 및 실험 경험**
- **논문 기반 기준을 구현 알고리즘으로 변환**
- **모델 재사용성 고려한 클래스 설계 및 패키징**

---

### 🚀 향후 개선 방향

- **머리카락/입술 색상까지 포함한 분석 정확도 향상**
- **진단 결과 기반 스타일 추천 기능 개발**
- **웹/모바일 서비스 연동 및 커머스 플랫폼과의 연결 검토**

---

### 🔗 참고 자료

- 논문: [퍼스널 컬러진단 가이드 요인간 상관관계와 뇌파분석](https://e-ajbc.org/journal/view.php?doi=10.20402/ajbc.2016.0071)
- 사용 모델: `dlib shape_predictor_68_face_landmarks.dat`

---
