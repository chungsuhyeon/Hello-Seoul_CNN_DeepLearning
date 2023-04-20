import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop

# CNN 모델 정의하기
def def_model(in_shape, nb_classes):
    model = Sequential()
    # Conv2D : 층을 추가하는 함수
    model.add(Conv2D(32,  # 32개의 커널 적용 = 32개의 필터라고 생각하면 됨
              kernel_size=(3, 3), # 커널의 크기
              activation='relu', # 사용할 활성화 함수
              input_shape=in_shape)) # 맨 처음층에 입력되는 값을 지정
    model.add(Conv2D(32, (3, 3), activation='relu'))
    
    # 정해진 구역 안에서 최댓값을 뽑아냄
    model.add(MaxPooling2D(pool_size=(2, 2))) # MaxPooling 창의 크기 2X2
    
    # 은닉층에 배치된 노드 중 일부를 임의로 끔 = 과적합 방지
    model.add(Dropout(0.25)) # 25%의 노드를 끔

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) # 벡터형태로 reshape = 2차원을 1차원으로 변경
    model.add(Dense(512, activation='relu')) # 은닉층 512개 / relu 사용
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='sigmoid')) # 출력층 nb_classes개 / softmax사용

    return model

# 컴파일하고 모델 반환하기
def get_model(in_shape, nb_classes):
    model = def_model(in_shape, nb_classes)

    # 최적화 함수 지정 = 모델 실행 환경 설정 (오차함수 : categorical_crossentropy / 최적화 함수 : rmsprop)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])

    # 최적화 함수 지정 = 모델 실행 환경 설정 (오차함수 : categorical_crossentropy / 최적화 함수 : rmsprop)
    model.compile(
        loss='mean_squared_error',
        # optimizer=RMSprop(),
        optimizer='adam',
        metrics=['accuracy'])

    print(model.summary())

    return model