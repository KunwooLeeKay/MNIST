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
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

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
# 4. Dense 레이어 조절
#   1) dense를 하나로 조정해보자. (노드 120짜리 남기고, 84짜리 삭제) -> 오히려 내려감.
#   2) 레이어의 전체 갯수는 그대로 유지하고, dense를 conv2d레이어로 바꿔보자. 음수되서 64짜리에 padding = same 추가

model =Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))) 
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), activation='relu')) 
model.add(MaxPooling2D())

model.add(Conv2D(64, (5,5), activation='relu', padding = 'same'))
model.add(MaxPooling2D())

model.add(Conv2D(128, (5,5), activation='relu', padding = 'same'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(120, activation='relu'))

model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True) 

history = model.fit(train_X, train_Y, epochs = 100, validation_data=(val_X, val_Y),callbacks=[early_stopping_cb])

model.evaluate(test_X, test_Y)

# 0.9876 -> 0.9902 확실한 업그레이드!
'''
Epoch 1/100
1688/1688 [==============================] - 20s 11ms/step - loss: 0.1755 - accuracy: 0.9489 - val_loss: 0.0776 - val_accuracy: 0.9767
Epoch 2/100
1688/1688 [==============================] - 19s 11ms/step - loss: 0.0724 - accuracy: 0.9790 - val_loss: 0.0781 - val_accuracy: 0.9790
Epoch 3/100
1688/1688 [==============================] - 19s 11ms/step - loss: 0.0567 - accuracy: 0.9835 - val_loss: 0.0618 - val_accuracy: 0.9818
Epoch 4/100
1688/1688 [==============================] - 18s 11ms/step - loss: 0.0472 - accuracy: 0.9867 - val_loss: 0.0514 - val_accuracy: 0.9858
Epoch 5/100
1688/1688 [==============================] - 18s 11ms/step - loss: 0.0415 - accuracy: 0.9883 - val_loss: 0.0531 - val_accuracy: 0.9862
Epoch 6/100
1688/1688 [==============================] - 19s 11ms/step - loss: 0.0411 - accuracy: 0.9886 - val_loss: 0.0436 - val_accuracy: 0.9888
Epoch 7/100
1688/1688 [==============================] - 20s 12ms/step - loss: 0.0360 - accuracy: 0.9899 - val_loss: 0.0423 - val_accuracy: 0.9897
Epoch 8/100
1688/1688 [==============================] - 20s 12ms/step - loss: 0.0346 - accuracy: 0.9911 - val_loss: 0.0494 - val_accuracy: 0.9892
Epoch 9/100
1688/1688 [==============================] - 18s 10ms/step - loss: 0.0336 - accuracy: 0.9914 - val_loss: 0.0704 - val_accuracy: 0.9842
Epoch 10/100
1688/1688 [==============================] - 18s 10ms/step - loss: 0.0278 - accuracy: 0.9925 - val_loss: 0.0442 - val_accuracy: 0.9890
Epoch 11/100
1688/1688 [==============================] - 19s 11ms/step - loss: 0.0363 - accuracy: 0.9911 - val_loss: 0.0819 - val_accuracy: 0.9823
Epoch 12/100
1688/1688 [==============================] - 19s 11ms/step - loss: 0.0231 - accuracy: 0.9938 - val_loss: 0.0583 - val_accuracy: 0.9910
Epoch 13/100
1688/1688 [==============================] - 18s 11ms/step - loss: 0.0305 - accuracy: 0.9925 - val_loss: 0.0559 - val_accuracy: 0.9880
Epoch 14/100
1688/1688 [==============================] - 17s 10ms/step - loss: 0.0284 - accuracy: 0.9929 - val_loss: 0.0569 - val_accuracy: 0.9920
Epoch 15/100
1688/1688 [==============================] - 18s 11ms/step - loss: 0.0228 - accuracy: 0.9946 - val_loss: 0.0869 - val_accuracy: 0.9875
Epoch 16/100
1688/1688 [==============================] - 17s 10ms/step - loss: 0.0255 - accuracy: 0.9942 - val_loss: 0.0550 - val_accuracy: 0.9883
Epoch 17/100
1688/1688 [==============================] - 17s 10ms/step - loss: 0.0302 - accuracy: 0.9933 - val_loss: 0.0893 - val_accuracy: 0.9877
313/313 [==============================] - 1s 3ms/step - loss: 0.0372 - accuracy: 0.9902
'''