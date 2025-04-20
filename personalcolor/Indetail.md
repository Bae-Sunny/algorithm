# 🎨 퍼스널 컬러 진단 알고리즘

## 📌 프로젝트 목표 및 개요

자신에게 어울리는 색상을 찾는 것은 많은 사람들의 관심사지만, 전문가의 진단은 비용과 시간이 소요됩니다. 이 프로젝트는 **컴퓨터 비전과 AI 기술**을 활용하여 사용자의 **얼굴 이미지로부터 피부색과 눈동자색을 객관적으로 추출**하고, 이를 기반으로 **개인에게 가장 적합한 퍼스널 컬러 시즌(봄/여름/가을/겨울)**을 과학적인 방법으로 진단하는 알고리즘을 개발하는 것을 목표로 합니다.

---

## ✨ 주요 기능 및 가치

- **과학적 컬러 분석:** dlib과 OpenCV를 활용하여 얼굴 이미지에서 **정확하게 피부와 눈동자 영역을 식별**하고, 해당 영역의 평균 RGB 값을 추출하여 개인의 고유한 컬러 특성을 객관적인 수치 데이터로 변환합니다.
- **논문 기반 시즌 분류:** 관련 연구 논문("Correlation between the Factors of Personal Color Diagnosis Guide and Brain Wave Analysis")에서 높은 상관관계를 보인 **피부색(70%)과 눈동자색(30%)에 가중치**를 부여하고, 4계절별 표준 RGB 값과의 **유클리드 거리를 계산**하여 가장 가능성이 높은 퍼스널 컬러 시즌을 과학적으로 진단합니다.
- **직관적인 결과 제공:** 분석된 피부색, 눈동자색과 함께 각 시즌별 확률을 **막대그래프로 시각화**하여 사용자가 자신의 퍼스널 컬러 진단 결과를 쉽고 명확하게 이해할 수 있도록 돕습니다.

---

## 🛠️ 기술 스택

- **Computer Vision & Image Processing:** Python, OpenCV (cv2), dlib
- **Data Analysis & Math:** NumPy, Scikit-learn (Euclidean Distance)
- **Visualization:** Matplotlib
- **ML Model Handling:** pickle
- **Development Environment:** Google Colab (Prototyping)
- *(Web Service Plan):* Django (Backend), Vue.js (Frontend), Highcharts (Web Visualization)

---

## 🧠 핵심 알고리즘 설계

1. **얼굴 및 랜드마크 감지:** `dlib`의 얼굴 탐지기와 68개 포인트 랜드마크 예측 모델을 사용하여 이미지에서 얼굴 영역과 주요 특징점(눈, 코, 입, 눈썹, 턱선 등)의 위치를 정확히 파악합니다.
2. **피부 영역 마스킹:** 얼굴 윤곽선 랜드마크를 이용해 전체 얼굴 영역 마스크를 생성한 후, 눈, 눈썹, 입 주변 랜드마크를 기반으로 해당 영역을 제외하여 순수한 피부 영역만 남도록 정밀한 마스크를 생성합니다.
3. **컬러 값 추출:**
    - **피부색:** 생성된 피부 마스크 영역 내 픽셀들의 평균 RGB 값을 계산합니다.
    - **눈동자색:** 눈 주변 랜드마크를 이용해 눈동자 영역을 추정하고, 해당 영역 픽셀들의 평균 RGB 값을 계산합니다. (좌우 평균 사용)
4. **퍼스널 컬러 시즌 분류:**
    - **기준값 설정:** 연구 논문을 참고하여 4계절(봄/여름/가을/겨울)별 대표적인 피부색과 눈동자색 RGB 표준값을 정의합니다.
    - **거리 계산:** 사용자의 추출된 피부색/눈동자색과 각 시즌별 표준 색상 값들 간의 유클리드 거리를 계산합니다. (각 시즌 내 여러 표준값 중 가장 가까운 거리 사용)
    - **가중 합산:** 피부색 거리(70%)와 눈동자색 거리(30%)에 설정된 가중치를 곱하여 합산, 각 시즌별 최종 거리를 계산합니다.
    - **최종 진단:** 계산된 최종 거리가 가장 짧은 시즌을 사용자의 퍼스널 컬러 시즌으로 진단합니다.
5. **시즌별 확률 계산:** 각 시즌별 최종 거리의 역수를 전체 역수 합으로 나누어, 각 시즌에 해당할 확률을 백분율로 계산합니다. 이를 통해 가장 가능성이 높은 시즌 외에 차선책도 제시할 수 있습니다.

---

## 🔄 구현 과정 하이라이트

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
    
   ![image](https://github.com/user-attachments/assets/95f91a0e-0b0d-4aa5-bbc8-7f0a470683c0)
   ![image](https://github.com/user-attachments/assets/a2639908-9854-42fe-9f05-9464b7016a58)


    

---

## 📊 테스트 결과 및 분석

다양한 인물 사진으로 테스트를 진행한 결과, 알고리즘은 각 이미지의 주요 색상 특징(따뜻함/차가움, 밝기, 채도 등)을 기반으로 퍼스널 컬러 시즌을 일관성 있게 분류했습니다.

![image](https://github.com/user-attachments/assets/0bda8dce-b212-46fa-aeaf-9985fddb55a5)


| 테스트 이미지 | 주요 특징 | 진단 결과 | 확률 분포 |
| --- | --- | --- | --- |
| 이미지 1 | 따뜻하고 밝은 톤, 밝은 눈동자 | Spring | Spring > Autumn > ... |
| 이미지 2 | 차갑고 부드러운 톤, 어두운 눈동자 | Summer | Summer > Winter > ... |
| 이미지 3 | 따뜻하고 차분한 톤, 갈색 눈동자 | Autumn | Autumn > Spring > ... |
| 이미지 4 | 차갑고 선명한 대비 | Winter | Winter > Summer > ... |

확률 분포 시각화를 통해 사용자는 가장 가능성이 높은 시즌뿐만 아니라, 다른 시즌과의 유사성 정도도 파악할 수 있어 진단 결과의 신뢰도를 높였습니다.

---

## 💡 주요 기술적 도전과 해결 과정

- **도전 1: 조명 및 각도 변화에 따른 랜드마크 불안정성**
    - **해결:** 이미지 전처리(밝기/대비 조절 등) 단계를 추가하고, dlib의 고성능 랜드마크 예측 모델(`shape_predictor_68_face_landmarks.dat`)을 사용하여 다양한 조건에서도 비교적 안정적인 랜드마크 좌표를 확보했습니다.
- **도전 2: 피부색 추출 시 얼굴 요소 간섭 문제**
    - **해결:** 눈, 눈썹, 입 주변 랜드마크를 정확히 파악하고 이를 제외하는 **정교한 마스킹 로직**을 개발하여 순수한 피부 영역의 평균 색상 값만을 추출하도록 개선했습니다.
- **도전 3: 객관적인 퍼스널 컬러 기준 부재**
    - **해결:** 관련 **연구 논문을 근거**로 피부색과 눈동자색의 중요도를 설정하고, 각 시즌별 대표 색상 RGB 값을 정의하여 **과학적이고 일관된 분류 기준**을 마련했습니다. 유클리드 거리와 가중치 시스템을 도입하여 객관성을 높였습니다.

---

## 📦 모델 패키징 및 향후 웹 서비스 구현 계획

개발된 퍼스널 컬러 분석 로직은 **Python 클래스(`PersonalColorAnalyzer`)로 구조화**하여 재사용성과 유지보수성을 높였습니다. 분석에 필요한 모델(dlib)과 로직을 포함한 클래스 인스턴스를 `pickle` 라이브러리를 사용하여 파일(`.pkl`)로 저장함으로써, 다른 환경이나 애플리케이션에서 쉽게 로드하여 사용할 수 있도록 했습니다.

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

**향후 계획:**

- **백엔드(Django):** 이미지 업로드 처리, 저장된 `.pkl` 모델 로드 및 분석 실행, 분석 결과(시즌, 확률, 색상값 등)를 JSON 형태로 반환하는 API를 개발합니다.
- **프론트엔드(Vue.js):** 사용자가 이미지를 쉽게 업로드하고, 분석 결과를 직관적으로 확인할 수 있는 웹 인터페이스를 개발합니다. 분석된 색상과 시즌별 확률 그래프는 **Highcharts** 와 같은 라이브러리를 활용하여 시각적으로 표현할 계획입니다.

![image](https://github.com/user-attachments/assets/48d2dfbe-fc3b-48b1-bc8b-cc9cb637a937)


---

## 🌱 성장 및 배움 (Key Takeaways)

- **컴퓨터 비전 기술 활용 능력:** OpenCV와 dlib을 이용한 얼굴 감지, 랜드마크 추출, 이미지 마스킹 등 **핵심적인 컴퓨터 비전 기술**을 실제 문제 해결에 적용하는 경험을 쌓았습니다.
- **데이터 기반 알고리즘 설계:** 연구 자료를 바탕으로 객관적인 기준을 설정하고, **수학적 계산(유클리드 거리, 가중 평균)**을 통해 복잡한 문제를 해결하는 알고리즘 설계 능력을 길렀습니다.
- **결과 시각화의 중요성:** 분석 결과를 사용자에게 효과적으로 전달하기 위한 **데이터 시각화 기법(Matplotlib 활용)**의 중요성을 배우고 실습했습니다.
- **모델 패키징 및 재사용성:** 개발한 분석 로직을 클래스로 구조화하고 `pickle`로 저장하는 과정을 통해 **모델의 재사용성과 배포 용이성**을 높이는 방법을 익혔습니다.

### 🚀 향후 개선 방향

1. **분석 정확도 향상:** 더 많은 인종과 다양한 조명 조건의 이미지 데이터셋을 구축하여 알고리즘의 강인성(Robustness)을 높이고, 머리카락 색상 등 추가적인 특징을 분석에 반영합니다.
2. **기능 확장:** 진단된 퍼스널 컬러에 맞는 **구체적인 색상 팔레트, 패션/메이크업 스타일 추천** 기능을 추가하고, 사용자가 직접 컬러 칩을 선택하여 비교해볼 수 있는 인터랙티브 기능을 개발합니다.
3. **플랫폼 연동:** 개발된 알고리즘을 **웹/모바일 애플리케이션**으로 확장하고, 패션 또는 뷰티 커머스 플랫폼과 연계하여 시너지 효과를 창출합니다.

---

## 🔗 관련 자료 및 링크

- **참고 논문:** "Correlation between the Factors of Personal Color Diagnosis Guide and Brain Wave Analysis" ([링크](https://e-ajbc.org/journal/view.php?doi=10.20402/ajbc.2016.0071))
- **사용한 모델:** dlib shape_predictor_68_face_landmarks.dat

---
