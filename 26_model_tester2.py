
from tensorflow import keras
import numpy as np

# 1. Load
mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test_original, Y_test_original) = mnist.load_data()
 
# 2. CNN은 채널 필요
X_train_full, X_test_original = X_train_full.reshape(-1, 28, 28, 1), X_test_original.reshape(-1, 28, 28, 1)
 
# 3. 보통 CNN은 zero-centering만 해준다.
    # per-channel pixel mean(VGGNet)을 빼거나 그냥 전체 mean(AlexNet)을 빼주면 되는데, MNIST는 채널이 1개라서 상관 없다.
    # normalize 안함 : 이미지 데이터에서 feature의 범위가 달라서 모델에 악영향을 미치지 않기 떄문.
    # PCA 안함 : 상관성 분석 필요 없음
    # Whitening 안함 : whitening : feature의 상관성을 없애주고 분산을 1로 만드는 것. 필요없음
 
X_train_full, X_test_original = X_train_full/1.0, X_test_original/1.0 # float로 변경해줌
avg = np.mean(X_train_full, axis = 0)
X_train_full -= avg
    # training data의 mean value를 다 빼주면서 zero-centering
X_test_original -= avg
 

nets = 10

model1 = []
model2 = []
results1, results2 = np.zeros( (X_test_original.shape[0],10) ), np.zeros( (X_test_original.shape[0],10) )

for i in range(nets):
    model1.append(keras.models.load_model\
        (r"C:\Users\user\Desktop\MNIST_project\models\FINAL_MODEL\trial1\MNIST_Final_"+str(i)+".h5"))
    model2.append(keras.models.load_model\
        (r"C:\Users\user\Desktop\MNIST_project\models\FINAL_MODEL\trial2\MNIST_Final_"+str(i)+".h5"))
    
for i in range(nets):
    results1 = results1 + model1[i].predict(X_test_original)
    results2 = results2 + model2[i].predict(X_test_original)



# Reverse to_categorical
actual = Y_test_original
prediction1 = np.argmax(results1, axis = 1)
prediction2 = np.argmax(results2, axis = 1)

correct1 = 0
wrong_indices1 = []
wrong_pred1 = []

correct2 = 0
wrong_indices2 = []
wrong_pred2 = []

for i in range(actual.shape[0]):
    if Y_test_original[i] == prediction1[i]:
        correct1 += 1
    else:
        wrong_indices1.append(i)
        wrong_pred1.append(prediction1[i])


for i in range(actual.shape[0]):
    if Y_test_original[i] == prediction2[i]:
        correct2 += 1
    else:
        wrong_indices2.append(i)
        wrong_pred2.append(prediction2[i])

accuracy1 = correct1/10000
accuracy2 = correct2/10000


 
print("="*50)
print("Final Ensembled Score : Accuracy = {0:.5f}, {0:.5f}".format(accuracy1, accuracy2))

print("===Mistaken Classes===")

r1, r2 = [], []

for i in range(len(wrong_indices1)):
    r1.append(Y_test_original[wrong_indices1[i]])

for i in range(len(wrong_indices2)):
    r2.append(Y_test_original[wrong_indices2[i]])

r1.sort(); r2.sort()

print(r1)
print(r2)

d1 = {};d2 = {}
d3 = {};d4 = {}

for i in range(len(wrong_indices1)):
    d1[wrong_indices1[i]] = Y_test_original[wrong_indices1[i]]
    d3[wrong_pred1[i]] = Y_test_original[wrong_indices1[i]]


for i in range(len(wrong_indices2)):
    d2[wrong_indices2[i]] = Y_test_original[wrong_indices2[i]]
    d4[wrong_pred2[i]] = Y_test_original[wrong_indices2[i]]

d1 = dict(sorted(d1.items(), key=lambda item: item[1]))
d2 = dict(sorted(d2.items(), key=lambda item: item[1]))
d3 = dict(sorted(d3.items(), key=lambda item: item[1]))
d4 = dict(sorted(d4.items(), key=lambda item: item[1]))

wrong_indices1 = list(d1.keys())
wrong_indices2 = list(d2.keys())

wrong_pred1 = list(d3.keys())
wrong_pred2 = list(d4.keys())


from matplotlib import pyplot as plt
print(wrong_indices1)
plt.figure(figsize=(15,10))
num = 0
X_test_original += avg # 가독성을 위해 zero-centering 제거

for i in range(len(wrong_indices1)):
    plt.subplot(5, 10, i+1)
    plt.imshow(X_test_original[wrong_indices1[i]].reshape((28,28)),cmap=plt.cm.binary)
    plt.title("pred = "+ str(wrong_pred1[i])+"\nans = "+str(Y_test_original[wrong_indices1[i]]), y=1.0)
    plt.axis('off')
    num += 1

plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.savefig("Mistaken1.jpg")

quit()
print(wrong_indices2)
plt.figure(figsize=(15,10))
num = 0

for i in wrong_indices2:
    plt.subplot(5, 10, num+1)
    plt.imshow(X_test_original[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.title("pred = "+ str(wrong_pred2[num])+"\nans = "+str(Y_test_original[i]), y=1.0)
    plt.axis('off')
    num += 1

plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.savefig("Mistaken2.jpg")