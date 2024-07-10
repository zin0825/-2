import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
print(tf.__version__)

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim =1))   #인풋 한덩어리, 아웃풋 한덩어리

#3. 컴파일, 훈련
model.compile (loss='mse', optimizer = 'adam')   #컴퓨터가 알아먹게 컴파일 한다.
model.fit(x, y, epochs=1000)   #fit은 훈련하다. x와 y를 1번 훈련시켜라.

#4. 평가, 예측
result=model.predict (np.array([4]))
print("4의 예측값 :", result)

