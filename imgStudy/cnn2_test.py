# CNN으로 MNIST 분류 문제 풀기
import cnn_model_test
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# 입력과 출력 지정하기
im_rows = 64 # 이미지의 높이
im_cols = 64 # 이미지의 너비
im_color = 3 # 이미지의 색공간 / 3 = 컬러
in_shape = (im_rows, im_cols, im_color) # 64X64 크기의 컬러 이미지
nb_classes = len(["김밥", "떡볶이", "불고기", "비빔밥", "삼겹살", "치킨"])

# 사진 데이터 읽어 들이기
photos = np.load('photos_64px.npz')
x = photos['x']
y = photos['y']

# 읽어들인 데이터를 3차원 배열로 변환하기
x = x.reshape(-1, im_rows, im_cols, im_color)
x = x.astype('float32') / 255     # 정규화
# 레이블 데이터를 One-hot 형식으로 변환하기 = 정답을 담음
y = keras.utils.np_utils.to_categorical(y.astype('int32'), nb_classes)

# 학습 전용과 테스트 전용으로 구분하기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

# 학습 전용 데이터 수 늘리기 --- (*1)
x_new = []
y_new = []
for i, xi in enumerate(x_train):
    yi = y_train[i]
    for ang in range(-30, 30, 5):
        # 회전 시키기 --- (*2)
        center = (16, 16) # 회전 중심
        mtx = cv2.getRotationMatrix2D(center, ang, 1.0)
        xi2 = cv2.warpAffine(xi, mtx, (0, 0)) # 이미지 이동
        x_new.append(xi2)
        y_new.append(yi)
        # 좌우 반전 --- (*3)
        xi3 = cv2.flip(xi2, 1)
        x_new.append(xi3)
        y_new.append(yi)

# 이미지를 늘린 데이터를 학습 데이터로 사용하기
print('수량을 늘리기 전=', len(y_train))
x_train = np.array(x_new)
y_train = np.array(y_new)
print('수량을 늘린 후=', len(y_train))

# CNN 모델 만들기 --- (*6)
model = cnn_model_test.get_model(in_shape, nb_classes)

# 학습 실행하기 --- (*8)
hist = model.fit(x_train, y_train,
          batch_size=64, # 몇개의 샘플로 가중치를 갱신할 것인지 = 전체 중 1/64문제를 풀때마다 해답을 맞춰보면서 가중치 갱신
          epochs=20, # 학습의 반복 횟수 = 너무 많이하면 오버피팅
          verbose=1,
          validation_data=(x_test, y_test))

# 모델 평가하기 --- (*9)
score = model.evaluate(x_test, y_test, verbose=1)
print('정답률=', score[1], 'loss=', score[0])
# accuracy는 값이 1에 가깝고 높을수록 좋은 모델임
# loss는 예측값과 실제값의 차이 = 값이 0에 가깝고 작을수록 좋음

# 학습 상태를 그래프로 그리기 --- (*10)
# 정답률 추이를 그래프로 그리기
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 손실 추이를 그래프로 그리기
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save_weights('./photos-model_add_layer.hdf5')
