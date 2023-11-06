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
# 2. conv 레이어를 하나 늘려보자. 32 -> 64 -> 128을 그냥하면 크기땜에 음수되서 padding = same까지 넣어준다. 
# 3. conv 레이어를 하나 늘려보자. 32 -> 64 -> 128 - > 256. 


model =Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))) 
model.add(MaxPooling2D())

model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(128, (5,5), activation='relu', padding = 'same'))
model.add(MaxPooling2D())

model.add(Conv2D(256, (5,5), activation='relu', padding = 'same'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(120, activation='relu'))

model.add(Dense(84, activation='relu'))

model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True) 

history = model.fit(train_X, train_Y, epochs = 100, validation_data=(val_X, val_Y),callbacks=[early_stopping_cb])

model.evaluate(test_X, test_Y)

# 0.9876 -> 0.9876 훈련 시간은 두배로 늘었는데 성능은 똑같다. 추가는 안하는게 낫다.
'''
Epoch 1/100
1688/1688 [==============================] - 69s 41ms/step - loss: 0.2119 - accuracy: 0.9386 - val_loss: 0.0881 - val_accuracy: 0.9750
Epoch 2/100
1688/1688 [==============================] - 72s 43ms/step - loss: 0.0901 - accuracy: 0.9749 - val_loss: 0.0794 - val_accuracy: 0.9788
Epoch 3/100
1688/1688 [==============================] - 71s 42ms/step - loss: 0.0738 - accuracy: 0.9801 - val_loss: 0.0868 - val_accuracy: 0.9790
Epoch 4/100
1688/1688 [==============================] - 74s 44ms/step - loss: 0.0602 - accuracy: 0.9829 - val_loss: 0.0894 - val_accuracy: 0.9785
Epoch 5/100
1688/1688 [==============================] - 74s 44ms/step - loss: 0.0618 - accuracy: 0.9839 - val_loss: 0.1248 - val_accuracy: 0.9695
Epoch 6/100
1688/1688 [==============================] - 71s 42ms/step - loss: 0.0537 - accuracy: 0.9862 - val_loss: 0.0584 - val_accuracy: 0.9853
Epoch 7/100
1688/1688 [==============================] - 70s 42ms/step - loss: 0.0447 - accuracy: 0.9882 - val_loss: 0.0503 - val_accuracy: 0.9885
Epoch 8/100
1688/1688 [==============================] - 71s 42ms/step - loss: 0.0474 - accuracy: 0.9887 - val_loss: 0.0781 - val_accuracy: 0.9855
Epoch 9/100
1688/1688 [==============================] - 69s 41ms/step - loss: 0.0529 - accuracy: 0.9877 - val_loss: 0.0651 - val_accuracy: 0.9865
Epoch 10/100
1688/1688 [==============================] - 69s 41ms/step - loss: 0.0472 - accuracy: 0.9888 - val_loss: 0.0657 - val_accuracy: 0.9853
Epoch 11/100
1688/1688 [==============================] - 74s 44ms/step - loss: 0.0364 - accuracy: 0.9907 - val_loss: 0.0538 - val_accuracy: 0.9893
Epoch 12/100
1688/1688 [==============================] - 76s 45ms/step - loss: 0.0390 - accuracy: 0.9905 - val_loss: 0.0704 - val_accuracy: 0.9848
Epoch 13/100
1688/1688 [==============================] - 75s 45ms/step - loss: 0.0315 - accuracy: 0.9920 - val_loss: 0.0540 - val_accuracy: 0.9885
Epoch 14/100
1688/1688 [==============================] - 75s 44ms/step - loss: 0.0393 - accuracy: 0.9914 - val_loss: 0.0663 - val_accuracy: 0.9872
Epoch 15/100
1688/1688 [==============================] - 72s 43ms/step - loss: 0.0377 - accuracy: 0.9922 - val_loss: 0.0881 - val_accuracy: 0.9875
Epoch 16/100
1688/1688 [==============================] - 72s 43ms/step - loss: 0.0320 - accuracy: 0.9928 - val_loss: 0.0577 - val_accuracy: 0.9892
Epoch 17/100
1688/1688 [==============================] - 72s 43ms/step - loss: 0.0271 - accuracy: 0.9938 - val_loss: 0.0902 - val_accuracy: 0.9873
313/313 [==============================] - 2s 5ms/step - loss: 0.0476 - accuracy: 0.9876
'''