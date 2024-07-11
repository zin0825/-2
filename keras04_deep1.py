from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# [실습] 레이어의 깊이와 노드의 갯수를 이용해서 [6]을 맹그러
# 에포는 100으로 고정, 건들지말것!!!
# 소수 네째자리까지 맞추면 합격. 예 : 6.0000 또는 5.9999

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(7, input_dim=8))
model.add(Dense(3, input_dim=7))
model.add(Dense(1, input_dim=3))


epochs = 100
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("==========================")
print("epochs : ", epochs)
print("로스 : ", loss)
result = model.predict([6])
print("6의 예측값 : ", result)



# epochs :  100
# 로스 :  0.0013291656505316496
# 6의 예측값 :  [[5.942788]]