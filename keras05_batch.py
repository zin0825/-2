from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])


# [실습] keras04의 가장 좋은 레이어와 노드를 이용하여,
# 최소의 loss를 맹그러
# batch_size 조절
# 에포는 100으로 고정을 풀어주겠노라!!!
# 로스 기준 0.32 미만!!!

#2. 모델구성
model = Sequential ()
model.add(Dense(4, input_dim=1))
model.add(Dense(7, input_dim=8))
model.add(Dense(3, input_dim=7))
model.add(Dense(3, input_dim=7))
model.add(Dense(1, input_dim=3))

epochs = 100
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs, batch_size=6)


#4. 평가, 예측
loss = model.evaluate(x,y)
print("=========================")
print("epochs : ", epochs)
print("로스 : ", loss)
result = model.predict([6])
print("6의 예측값 : ", result)


# epochs :  100
# 로스 :  0.32554206252098083
# 6의 예측값 :  [[5.790636]]