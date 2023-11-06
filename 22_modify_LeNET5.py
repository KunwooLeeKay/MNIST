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
from keras.preprocessing.image import ImageDataGenerator

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


# 6. Data Augmentation

datagen = ImageDataGenerator(
    rotation_range=15, # 대충 1을 한 15도 정도까지 기울여서 써도 알아보니까 15도로 설정 
    width_shift_range=0.1,
    height_shift_range=0.1, # 너무 크게하면 7이랑 1이랑 너무 비슷할듯 하니까 작게 설정
    shear_range=0.0, # 삐뚜로 쓰는것도 넣으면 rotation이랑 겹쳐서 너무 왜곡될 수 있으니 안한다.
    zoom_range=0.1, # 크기 조절
)
datagen.fit(train_X)

############## MODELS ##################

# 1. LeNET은 필터 갯수 6, 16을 사용하였는데, 요새 대부분의 코드는 32, 64 ,128 이런식으로 간다. 바꿔보자.
# 2. conv 레이어를 하나 늘려보자. 32 -> 64 -> 128을 그냥하면 크기땜에 음수되서 128에 padding = same까지 넣어준다. 
# 3. (deleted) conv 레이어를 하나 늘려보자. 32 -> 64 -> 128 - > 256. : 안하는게 낫다!!
# 4. dense를 하나로 조정해보자. (노드 120짜리 남기고, 84짜리 삭제) -> 오히려 내려감.
# 5. 레이어의 전체 갯수는 그대로 유지하고, dense를 conv2d레이어로 바꿔보자. 음수되서 64짜리에 padding = same 추가
# 6. Dense를 없앤게 확실히 성능이 좋아졌다. 하나 더 없애보자... 이 이상으론 못없앰. 너무 작아져가지구
# 7. BatchNormalization 추가 : 확실한 업그레이드
# 8. Dropout 추가 - 성능이 떨어짐 : 제거
# 8. Data Augmentation 추가 : 굳

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

history = model.fit(datagen.flow(train_X, train_Y), epochs = 100, validation_data=(val_X, val_Y),callbacks=[early_stopping_cb])

model.evaluate(test_X, test_Y)

# 0.9928 -> 0.9953 확실한 업그레이드
'''
Epoch 1/100
1688/1688 [==============================] - 25s 14ms/step - loss: 0.2096 - accuracy: 0.9342 - val_loss: 0.1150 - val_accuracy: 0.9643
Epoch 2/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0888 - accuracy: 0.9726 - val_loss: 0.0635 - val_accuracy: 0.9818
Epoch 3/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.0719 - accuracy: 0.9775 - val_loss: 0.0378 - val_accuracy: 0.9872
Epoch 4/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0614 - accuracy: 0.9810 - val_loss: 0.0397 - val_accuracy: 0.9882
Epoch 5/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0540 - accuracy: 0.9836 - val_loss: 0.0772 - val_accuracy: 0.9763
Epoch 6/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0505 - accuracy: 0.9844 - val_loss: 0.0424 - val_accuracy: 0.9895
Epoch 7/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0447 - accuracy: 0.9867 - val_loss: 0.0270 - val_accuracy: 0.9922
Epoch 8/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0439 - accuracy: 0.9866 - val_loss: 0.0360 - val_accuracy: 0.9897
Epoch 9/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0404 - accuracy: 0.9871 - val_loss: 0.0343 - val_accuracy: 0.9900
Epoch 10/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0384 - accuracy: 0.9882 - val_loss: 0.0261 - val_accuracy: 0.9923
Epoch 11/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0368 - accuracy: 0.9889 - val_loss: 0.0442 - val_accuracy: 0.9892
Epoch 12/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0360 - accuracy: 0.9889 - val_loss: 0.0311 - val_accuracy: 0.9908
Epoch 13/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0339 - accuracy: 0.9897 - val_loss: 0.0199 - val_accuracy: 0.9938
Epoch 14/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.0310 - accuracy: 0.9910 - val_loss: 0.0395 - val_accuracy: 0.9878
Epoch 15/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0302 - accuracy: 0.9906 - val_loss: 0.0355 - val_accuracy: 0.9907
Epoch 16/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.0287 - accuracy: 0.9911 - val_loss: 0.0256 - val_accuracy: 0.9925
Epoch 17/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0287 - accuracy: 0.9913 - val_loss: 0.0309 - val_accuracy: 0.9907
Epoch 18/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.0267 - accuracy: 0.9915 - val_loss: 0.0341 - val_accuracy: 0.9897
Epoch 19/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.0278 - accuracy: 0.9917 - val_loss: 0.0314 - val_accuracy: 0.9912
Epoch 20/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0254 - accuracy: 0.9923 - val_loss: 0.0278 - val_accuracy: 0.9915
Epoch 21/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.0257 - accuracy: 0.9920 - val_loss: 0.0264 - val_accuracy: 0.9920
Epoch 22/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.0231 - accuracy: 0.9927 - val_loss: 0.0273 - val_accuracy: 0.9923
Epoch 23/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0235 - accuracy: 0.9927 - val_loss: 0.0279 - val_accuracy: 0.9923
313/313 [==============================] - 1s 5ms/step - loss: 0.0178 - accuracy: 0.9953
'''