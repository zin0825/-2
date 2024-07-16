# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas  as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = "./_data/따릉이/"   # path = 경로/ 같은 경로를 한번에

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# 함수 csv파일을 불러들이겠다, .은 루트, ""은 문자 "1"+"A" = "1A" 그냥 붙여버림
print(train_csv)   # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)   # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)   # [715 rows x 1 columns]

print(train_csv.shape)   # (1459, 10)
print(test_csv.shape)   # (715, 9)
print(submission_csv.shape)   # (715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64

################### 결측치 처리 1. 삭제 ####################
# print(train_csv.isnull().sum())   # 결측치가 있냐
# hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117
# count                       0

print(train_csv.isna().sum())
# hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117
# count                       0

train_csv = train_csv.dropna()   # 데이터를 떨군다, 삭제
print(train_csv.isna().sum())   # 결측치가 있니, 결측치 없어요, 결측치를 전부 더해줘
# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# count                     0

print(train_csv)   # [1328 rows x 10 columns]
print(train_csv.isna().sum())
# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# count                     0
print(train_csv.info())
#  0   hour                    1328 non-null   int64
#  1   hour_bef_temperature    1328 non-null   float64
#  2   hour_bef_precipitation  1328 non-null   float64
#  3   hour_bef_windspeed      1328 non-null   float64
#  4   hour_bef_humidity       1328 non-null   float64
#  5   hour_bef_visibility     1328 non-null   float64
#  6   hour_bef_ozone          1328 non-null   float64
#  7   hour_bef_pm10           1328 non-null   float64
#  8   hour_bef_pm2.5          1328 non-null   float64
#  9   count                   1328 non-null   float64

print(test_csv.info())   
# 훈련 안하고 평가만 할 것, 결측치를 삭제할 수 없음 서브미션에 넣어야하기 때문에, 
# 결측치에 평균값을 넣자
#  0   hour                    715 non-null    int64
#  1   hour_bef_temperature    714 non-null    float64
#  2   hour_bef_precipitation  714 non-null    float64
#  3   hour_bef_windspeed      714 non-null    float64
#  4   hour_bef_humidity       714 non-null    float64
#  5   hour_bef_visibility     714 non-null    float64
#  6   hour_bef_ozone          680 non-null    float64
#  7   hour_bef_pm10           678 non-null    float64
#  8   hour_bef_pm2.5          679 non-null    float64

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())
#  0   hour                    715 non-null    int64
#  1   hour_bef_temperature    715 non-null    float64
#  2   hour_bef_precipitation  715 non-null    float64
#  3   hour_bef_windspeed      715 non-null    float64
#  4   hour_bef_humidity       715 non-null    float64
#  5   hour_bef_visibility     715 non-null    float64
#  6   hour_bef_ozone          715 non-null    float64
#  7   hour_bef_pm10           715 non-null    float64
#  8   hour_bef_pm2.5          715 non-null    float64

x = train_csv.drop(['count'], axis=1)   # .drop 컬럼 하나를 삭제 / axis = 중심서느 행과 열 따라 동작
print(x)   # [1328 rows x 9 columns]
y = train_csv['count']   # train_csv만 넣어줘
print(y)
# id
# 3        49.0
# 6       159.0
# 7        26.0
# 8        57.0
# 9       431.0
#         ...
# 2174     21.0
# 2175     20.0
# 2176     22.0
# 2178    216.0
# 2179    170.0
# Name: count, Length: 1328, dtype: float64

print(y.shape)   # (1328,)  / 여기까지가 전처리 부분

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=5757)

print(x)
print(y)
print(x.shape, y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(500, input_dim=9))
model.add(Dense(250))
model.add(Dense(125))
model.add(Dense(79))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)   # batch_size는 열

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)   # 여기는 훈련이 아님
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# 로스 :  2277.40234375
# r2스코어 :  0.6401915659983989
