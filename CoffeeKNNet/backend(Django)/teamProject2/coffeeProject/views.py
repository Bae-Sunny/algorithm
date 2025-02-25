import joblib
import requests
from django.http import JsonResponse
import pandas as pd
import json
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def coffeePredict(request):
    if request.method == 'POST':
        # POST 요청에서 JSON 데이터를 가져옴
        coffeeFormData = json.loads(request.body)
        print(f'받아온 데이터 : {coffeeFormData}')

        # 데이터를 DataFrame으로 변환하고 값들을 float 타입으로 변환
        new_data = pd.DataFrame([coffeeFormData]).astype('float')

        # 훈련된 모델이 저장된 경로
        model_path = '/home/ict/PycharmProject/teamProject2/coffeeProject/model'

        # 훈련된 모델을 로드
        loaded_model = joblib.load(f'{model_path}/voting_classifier.pkl')

        # 예측을 수행
        pred = loaded_model.predict(new_data)

        # JSON 응답을 반환
        return JsonResponse({'pred': float(pred[0])})
