from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. 데이터
x = np.array(range(1, 101))
# x2 = array(range(1, 101))
y = np.array(range(101, 201))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim = 1, activation='relu'))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1, 
            validation_data=(x_val, y_val))
            
#4. 예측, 평가
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)

y_predict = model.predict([101, 102, 103])
print("y_predict : ", y_predict)






