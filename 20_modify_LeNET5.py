# Usually CNN consists of convolution-subsampling pairs -> dense layer

# 1. How many pairs of conv-subs? -> try 1 ~ 3 (usually, complicated problems require more)
# 2. How many filters in conv? : usually increases (ex. 24 -> 48 -> 64 ...) -> try 8, 16 / 16, 32 / ...
# 3. How deep / large dense layer? : 1 layer 0, 32, 64, ... 2048 nodes / 2layer ...
# 4. How much dropouts?

# 5. Advanced : Include BatchNormalization?, replace '32C5' with '32C3-32C3', replace 'P2' with '32C5S2'
#               add batch normalization, add data augmentation

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization

# 1. Load
mnist = keras.datasets.mnist
(X_train_full, y_train_full), (test_X, test_Y) = mnist.load_data()

# 2. CNN은 채널 필요
X_train_full, test_X = X_train_full.reshape(-1, 28, 28, 1), test_X.reshape(-1, 28, 28, 1)
print(X_train_full.shape, test_X.shape)

# 3. 보통 CNN은 zero-centering만 해준다. 
    # per-channel pixel mean(VGGNet)을 빼거나 그냥 전체 mean(AlexNet)을 빼주면 되는데, MNIST는 채널이 1개라서 상관 없다.
    # normalize 안함 : 이미지 데이터에서 feature의 범위가 달라서 모델에 악영향을 미치지 않기 떄문.
    # PCA 안함 : 상관성 분석 필요 없음
    # Whitening 안함 : whitening : feature의 상관성을 없애주고 분산을 1로 만드는 것. 필요없음

X_train_full, test_X = X_train_full/1.0, test_X/1.0 # float로 변경해줌
avg = np.mean(X_train_full, axis = 0)
X_train_full -= avg
    # training data의 mean value를 다 빼주면서 zero-centering
test_X -= avg

# 4. Data Split
random_seed = 0
train_X, val_X, train_Y, val_Y = \
    train_test_split(X_train_full, y_train_full, random_state = random_seed, test_size = 0.1, stratify = y_train_full)

# 5. OneHot
train_Y, val_Y, test_Y = \
    to_categorical(train_Y, num_classes=10), to_categorical(val_Y, num_classes=10), to_categorical(test_Y, num_classes=10)


############## MODELS ##################

# 1. LeNET은 필터 갯수 6, 16을 사용하였는데, 요새 대부분의 코드는 32, 64 ,128 이런식으로 간다. 바꿔보자.
# 2. conv 레이어를 하나 늘려보자. 32 -> 64 -> 128을 그냥하면 크기땜에 음수되서 128에 padding = same까지 넣어준다. 
# 3. (deleted) conv 레이어를 하나 늘려보자. 32 -> 64 -> 128 - > 256. : 안하는게 낫다!!
# 4. dense를 하나로 조정해보자. (노드 120짜리 남기고, 84짜리 삭제) -> 오히려 내려감.
# 5. 레이어의 전체 갯수는 그대로 유지하고, dense를 conv2d레이어로 바꿔보자. 음수되서 64짜리에 padding = same 추가
# 5. Dense를 없앤게 확실히 성능이 좋아졌다. 하나 더 없애보자... 이 이상으론 못없앰. 너무 작아져가지구
# 6. BatchNormalization 추가 : 확실한 업그레이드

model =Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))) 
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), activation='relu')) 
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(64, (5,5), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(128, (5,5), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(120, activation='relu'))

model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True) 

history = model.fit(train_X, train_Y, epochs = 100, validation_data=(val_X, val_Y),callbacks=[early_stopping_cb])

model.evaluate(test_X, test_Y)

# 0.9902 -> 0.9928 
'''
Epoch 1/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.1144 - accuracy: 0.9640 - val_loss: 0.0535 - val_accuracy: 0.9833
Epoch 2/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0451 - accuracy: 0.9864 - val_loss: 0.0401 - val_accuracy: 0.9878
Epoch 3/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0341 - accuracy: 0.9892 - val_loss: 0.0385 - val_accuracy: 0.9895
Epoch 4/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0299 - accuracy: 0.9904 - val_loss: 0.0466 - val_accuracy: 0.9855
Epoch 5/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0234 - accuracy: 0.9925 - val_loss: 0.0380 - val_accuracy: 0.9902
Epoch 6/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0193 - accuracy: 0.9939 - val_loss: 0.0445 - val_accuracy: 0.9872
Epoch 7/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0170 - accuracy: 0.9945 - val_loss: 0.0327 - val_accuracy: 0.9898
Epoch 8/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0140 - accuracy: 0.9955 - val_loss: 0.0321 - val_accuracy: 0.9918
Epoch 9/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0105 - accuracy: 0.9968 - val_loss: 0.0337 - val_accuracy: 0.9912
Epoch 10/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0128 - accuracy: 0.9960 - val_loss: 0.0457 - val_accuracy: 0.9905
Epoch 11/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0088 - accuracy: 0.9973 - val_loss: 0.0390 - val_accuracy: 0.9915
Epoch 12/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0096 - accuracy: 0.9969 - val_loss: 0.0339 - val_accuracy: 0.9920
Epoch 13/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0085 - accuracy: 0.9973 - val_loss: 0.0325 - val_accuracy: 0.9922
Epoch 14/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0088 - accuracy: 0.9971 - val_loss: 0.0438 - val_accuracy: 0.9895
Epoch 15/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0066 - accuracy: 0.9978 - val_loss: 0.0373 - val_accuracy: 0.9935
Epoch 16/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0065 - accuracy: 0.9979 - val_loss: 0.0373 - val_accuracy: 0.9917
Epoch 17/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0065 - accuracy: 0.9983 - val_loss: 0.0455 - val_accuracy: 0.9885
Epoch 18/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0062 - accuracy: 0.9983 - val_loss: 0.0323 - val_accuracy: 0.9927
313/313 [==============================] - 1s 4ms/step - loss: 0.0310 - accuracy: 0.9928
'''