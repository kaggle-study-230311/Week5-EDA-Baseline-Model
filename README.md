# 5주차_EDA + Baseline Moldel(0.40 RMSE)

## 0. 문제정의

- 목표 : 뉴욕시에서 택시를 타고 이동하는 총 시간을 예측하는 모델을 구축
- 평가지표 : RMSLE(평균제곱로그오차)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c117943e-ae64-4cbd-802d-85d2ca7be204/Untitled.png)
    
    - $\epsilon$ 은 RMSLE 값(점수)입니다.
    - $n$ 은 (공개/비공개) 데이터 집합의 총 관측 횟수입니다,
    - $p_i$ 는 여행 기간에 대한 예측값, 그리고
    - $a_i$는 $i$에 대한 실제 여행 기간입니다.
    - $log(x)$는 $(x)$의 자연 로그입니다.
    - RMSLE(Root Mean Squared Log Error)?
        - 회귀 평가지표 중 하나(MAE, MSE, RMSE, MSLE, RMSLE, R^2)
            
            [머신러닝 - 17. 회귀 평가 지표](https://bkshin.tistory.com/entry/머신러닝-17-회귀-평가-지표)
            
        - 값이 작을수록 회귀 성능이 좋음 → 예측값과 실체값의 차이가 없다는 뜻
        - RMSE 대비 RMLSE의 장점
            - 아웃라이어가 있어도 값의 변동폭이 크지 않음 → RMSE에 비해 아웃라이어에 robust하다
                - 예측값 = 67, 78, 91, 실제값 = 60, 80, 90일 때 RMSE = 4.242, RMLSE = 0.6466
                - 예측값 = 67, 78, 91, 102, 실제값 = 60, 80, 90, 750일 때 RMSE = 374.724, RMSLE = 1.160
            - 상대적 Error를 측정해준다 → RMSE와 달리 예측값과 실제값의 상대적 Error를 측정해줌
                - 예측값 = 100, 실제값 = 90일 때, RMSLE = 0.1053, RMSE = 10
                - 예측값 = 10,000, 실제값 = 9,000일 때, RMSLE = 0.1053, RMSE = 1,000
            - Over Estimation보다 Under Estimation에 큰 패널티 부여
                - 예측값 = 600, 실제값 = 1,000일 때 RMSE = 400, RMSLE = 0.510
                - 예측값 = 1400, 실제값 = 1,000일 때 RMSE = 400, RMSLE = 0.33
- 타겟(id별 여행 소요시간 예측)
    - `id`, `trip_duration`
- 데이터셋
    - train
    
    ```python
    #   Column              Non-Null Count    Dtype  
    ---  ------              --------------    -----  
     0   id                  1458644 non-null  object 
     1   vendor_id           1458644 non-null  int64  
     2   pickup_datetime     1458644 non-null  object 
     3   dropoff_datetime    1458644 non-null  object 
     4   passenger_count     1458644 non-null  int64  
     5   pickup_longitude    1458644 non-null  float64
     6   pickup_latitude     1458644 non-null  float64
     7   dropoff_longitude   1458644 non-null  float64
     8   dropoff_latitude    1458644 non-null  float64
     9   store_and_fwd_flag  1458644 non-null  object 
     10  trip_duration       1458644 non-null  int64
    ```
    
    - test
    
    ```python
    #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   id                  625134 non-null  object 
     1   vendor_id           625134 non-null  int64  
     2   pickup_datetime     625134 non-null  object 
     3   passenger_count     625134 non-null  int64  
     4   pickup_longitude    625134 non-null  float64
     5   pickup_latitude     625134 non-null  float64
     6   dropoff_longitude   625134 non-null  float64
     7   dropoff_latitude    625134 non-null  float64
     8   store_and_fwd_flag  625134 non-null  object
    ```
    
    - columns
    
    ```python
    id - 각 여행의 고유 식별자
    vendor_id - 트립 기록과 연결된 공급업체를 나타내는 코드
    pickup_datetime - 미터기가 작동한 날짜 및 시간
    drop-off_datetime - 미터기가 연결 해제된 날짜 및 시간
    passenger_count - 차량에 탑승한 승객 수(드라이버가 입력한 값)
    pickup_longitude - 미터기가 작동된 경도
    pickup_latitude - 미터기가 작동된 위도
    dropoff_longitude - 미터기가 연결 해제된 경도
    dropoff_latitude - 미터기를 분리한 위도
    store_and_fwd_flag - 이 플래그는 차량이 서버에 연결되지 않았기 때문에 차량 메모리에 트립 기록이 저장된 후 공급업체로 전송되었는지 여부를 나타냅니다(Y=저장 및 전달, N=저장 및 전달 트립이 아님).
    trip_duration - 트립 기간(초)
    
    Disclaimer: 커널에서 사용할 수 있는 확장된 변수 집합을 제공하기 위해 데이터 세트 순서에서 하차 좌표를 제거하지 않기로 결정했습니다.
    ```
    

## 1. 세팅 및 데이터 확인

- import library
- read csv
- check shape
- null 값 체크
- 적절한 데이터 유형으로 변환
    - 날짜 데이터 컬럼을 datetime 객체로 변환
- train data의 기간 체크
    - min, max
- **`pickup_datetime`**에서 일 / 월 / 시간 분리
    - 요일별, 일별, 시간대별 분석 가능
- 위도 경도를 소수점 셋째 자리까지 반올림
    - 데이터의 정확도를 일부 떨어뜨리더라도 데이터의 크기를 줄이고 이후에 데이터 처리를 보다 효율적으로 수행하기 위한 목적
- 여행거리 계측
    - Haversine Distance
        
        [최단거리 구하기, 하버사인 공식(Haversine Formula)](https://kayuse88.github.io/haversine/)
        

## EDA

- 'trip_duration_in_hour' 열의 분포 시각화
- 이상치 체크
- 로그 변환을 통해 `Distribution of trip duration (sec)` 분포 확인 
→ 로그 변환은 치우친 분포를 정규 분포에 가깝게 만들어줌
- heatmap으로 전체적인 탐색
    - pickup 지역의 분포를 heatmap으로 시각화
    - train 데이터프레임의 pickup 지점의 위도와 경도를 산점도(scatter plot)로 시각화
    - dropoff 지점의 분포를 heatmap으로 시각화
    - pickup 지점에서의 평균 trip_duration을 heatmap으로 시각화
- 항목별 분포 확인
    - pickup 시간대(pickup_hour) 분포를 시각화
    - dropoff 시간대(dropoff_hour) 분포를 시각화
    - 전체 기간의(시계열) pickup 횟수 분포를 시각화
    - trip_distance(운행 거리)의 로그값 분포를 시각화
    - trip_distance(운행 거리)와 trip_duration(운행 시간) 간의 관계를 산점도로 시각화
    - pickup_hour(픽업 시간)별 평균 trip_duration(운행 시간)을 시각화
    - pickup_day_of_week(요일)을 기준으로 각 요일별 탑승 건수를 시각화
    - pickup_day_of_week(픽업 요일)별 평균 trip_duration(운행 시간)을 시각화
    - 위도(latitude)와 경도(longitude)를 이용하여 방위각(Bearing)을 계산
        - 베어링 컬럼을 추가
    - 방위각 분포도 시각화
    - 방위각(bearing) vs 운행 시간(trip duration) 산점도(Scatter Plot)
        - 이상값 체크
    - store_and_fwd_flag와 trip_duration(운행 시간) 간의 관계
- K-means Cluster
    - 클러스터화 된 픽업 위치 데이터를 각 클러스터에 매핑한 뒤, 지도상에 픽업 지역을 시각화
    - 클러스터별 픽업 수 분포
    - 클러스터 별 여행 기간
- 택시의 평균 속도
- 요일별(kph) 평균 속도 시각화

## Feature Engineering

- test data에 동일한 처리 진행

## Modeling

- 불필요한 컬럼 삭제
    - 'avg_speed_kph'는 'trip_distance'와 'trip_duration_in_hour' 컬럼을 기반으로 계산된 값이므로, 모델이 예측하는데 있어 중복적인 정보
    - 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude' 컬럼들은 이미 k-means 군집화를 통해 군집 번호로 변환되었기 때문에, 좌표 정보 자체는 필요하지 않음
- 'trip_duration' 컬럼을 로그 변환한 새로운 컬럼으로 대체하고, 기존의 'trip_duration' 컬럼은 삭제
    - 로그 스케일 이유는 앞서 언급
- 요일 인코딩 진행
- 훈련을 위한 FE를 끝낸 파일 저장 및 기존 변수 메모리 삭제
- def LabelEncoding(*train_df*,*test_df*,*max_levels*=2):
- def readInputAndEncode(*input_path*,*train_file*,*test_file*,*target_column*):
- def GetFeaturesAndSplit(*train*,*test*,*target*,*imputing_strategy*='median',*split*=0.25,*imputation*=True):
- 선형 회귀
    - 모델 생성 및 학습
    - 검증셋 예측
    - RMSE
    - 테스트셋 예측
- 랜덤 포레스트 회귀
    - 모델 생성 및 학습
    - 검증셋 예측
    - RMSE
    - 테스트셋 예측
