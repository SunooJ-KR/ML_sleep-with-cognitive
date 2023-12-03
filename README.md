# 수면의 질이 인지기능의 미치는 영향 분석

## 담당
|순서|이름|내용|예정기한|
|------|---|---|---|
|1|민정|데이터 전처리|11/22|
|2|선우|예측모델 구현|12/3|
|3|민정, 선우|논문 작성|12/7|

## 내용
### 활용 데이터
AI-Hub 에서 제공하는 [치매 고위험군 웨어러블 라이프로그](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=226) 데이터

**데이터 종류**
1. 라이프로그(수면정보)
2. 라이프로그(걸음거리 정보)
3. 인지기능(MMSE)

### 머신러닝 모델
- XGBoost Classifier
    파라미터 : n_estimators=500, learning_rate=0.3, max_depth=4, random_state=32


## 결과
### 1차 전처리 데이터셋
1. sleep
정확도:0.5172, 정밀도:0.3333, 재현율:0.2727, F1:0.3000, AUC:0.5303
2. activity

### 2차 전처리 데이터셋
```python
data_path = '../datasets/preprocessing/'
# cn : 정상 -> target컬럼 0으로 변환
# dem : 치매 -> target컬럼 1으로 변환
activity_cn_df = pd.read_csv(data_path+'activity_preprocessing_final_CN.csv').drop(['DIAG_NM','EMAIL'],axis=1)
activity_cn_df['target'] = [0 for i in range(len(activity_cn_df))]
activity_dem_df = pd.read_csv(data_path+'activity_preprocessing_final_Dem.csv').drop(['DIAG_NM','EMAIL'],axis=1)
activity_dem_df['target'] = [1 for i in range(len(activity_dem_df))]
# 활동 데이터 통합
activity_df = pd.concat([activity_cn_df, activity_dem_df], ignore_index=True)

sleep_cn_df = pd.read_csv(data_path+'sleep_preprocessing_final_CN.csv').drop(['DIAG_NM','EMAIL'],axis=1)
sleep_cn_df['target'] = [0 for i in range(len(sleep_cn_df))]
sleep_dem_df = pd.read_csv(data_path+'sleep_preprocessing_final_Dem.csv').drop(['DIAG_NM','EMAIL'],axis=1)
sleep_dem_df['target'] = [1 for i in range(len(sleep_dem_df))]
# 수면 데이터 통합
sleep_df = pd.concat([sleep_cn_df, sleep_dem_df], ignore_index=True)
```
통합된 데이터의 80%는 학습, 20%는 테스트 데이터로 활용되었음.\
또한, random하게 데이터를 비복원 추출하여 학습 및 테스트 데이터로 활용.

1. sleep
정확도:0.8047, 정밀도:0.7703, 재현율:0.6746, F1:0.7192, AUC:0.8713
2. activity
정확도:0.7059, 정밀도:0.6522, 재현율:0.6078, F1:0.6292, AUC:0.7813
