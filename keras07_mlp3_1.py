import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]])   # 2개이기에 매트릭스 형태

print(x.shape)   # (3, 10)
print(y.shape)   # (2, 10)

x = x.T   # 행렬 변경
y = np.transpose(y)   # y = y.T와 같은 원리, 최종 아웃풋 노드의 갯수
print(x.shape)   # (10, 3)
print(y.shape)   # (10, 2)


#2. 모델구성
# [실습] 맹그러봐
# x_predict = [10, 31, 211]

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])
print('로스 : ', loss)
print('[10,31,211]의  예측값 : ', result)

# 로스 :  0.3501274883747101
# [10,31,211]의  예측값 :  [[10.800976    0.38933513]]
