import numpy as np
import tensorflow as tf

#1. 데이터 -> 정제된 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size  = 3)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size = 3)
print('loss : ', loss)


results = model.predict([4])
print('results : ', results)


# 과제 1. mse
# 과제 2. batch_size의 디폴트 값
# 과제 3. 하이퍼파라이터 튠을 마구마구 해볼것, batch_size랑 노드, epoch 등 바꿔보기

# kingkeras@naver.com 으로 제출
