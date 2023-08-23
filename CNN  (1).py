#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow
import pandas as pd


# In[18]:


train=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\train.csv")
test=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\test.csv")


# In[19]:


EPOCHS = 50
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root = r'G:\capstone\project\Apple_Trees/plant_images'
width=256
height=256
depth=3

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")


# In[20]:


image_size = len(image_list)

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

print(label_binarizer.classes_)

np_image_list = np.array(image_list, dtype=np.float16) / 225.0

print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42)

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")

model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


# In[21]:


model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])


# In[22]:


# train the network
print("[INFO] training network...")

history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")


# # Classification report using keras

# In[23]:


from keras import metrics
import keras


# In[24]:


model.compile(loss='mean_squared_error', optimizer='sgd',
              metrics=[metrics.mae,
                       metrics.categorical_accuracy])


# In[25]:


import seaborn as sns


# In[26]:


from sklearn.preprocessing import LabelEncoder


# In[27]:


le =LabelEncoder()


# In[28]:


train['image_id']= le.fit_transform(train['image_id'])


# In[29]:


sns.heatmap(train, vmin=50, vmax=100)


# In[30]:


sns.heatmap(y_train, annot=True,xticklabels=y_test, yticklabels=y_test)


# In[31]:


model.compile(metrics=[1,224,224,3])


# In[32]:


def recall(test_image, result):
    y_true = K.ones_like(test_image)
    true_positives = K.sum(K.round(K.clip(test_image * result, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


# In[33]:


def precision(test_image, result):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(test_image * result, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(test_image, result):
    precision = precision_m(test_image, result)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[34]:


model.compile(metrics=['accuracy', f1_score, precision, recall])


# In[35]:


x_train.shape


# In[36]:


y_train.shape


# In[37]:


x_test.shape


# In[38]:


y_test.shape


# In[ ]:





# In[39]:


from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[40]:


le =LabelEncoder()


# In[41]:


train['image_id']= le.fit_transform(train['image_id'])


# In[42]:


train['image_id'].unique()


# In[43]:


le =LabelEncoder()


# In[44]:


test['image_id']= le.fit_transform(test['image_id'])


# In[45]:


test['image_id'].unique()


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42)


# In[47]:


logmodel = tree.DecisionTreeClassifier()


# In[48]:


logmodel.fit(X_train, y_train)


# In[49]:


predictions = logmodel.predict(X_test)


# # Classification report

# In[50]:


classification_report(y_test,predictions)


# In[51]:


print(confusion_matrix(y_test, predictions))


# In[ ]:





# In[52]:


classifier_tree = tree.DecisionTreeClassifier()


# In[53]:


y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[54]:


print(classification_report(y_test, y_predict))


# In[ ]:





# In[55]:


import pandas as pd
X=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\train.csv")
y=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\test.csv")


# In[56]:


X=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\train.csv")
y=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\test.csv")


# In[57]:


from sklearn.preprocessing import LabelEncoder


# In[58]:


le =LabelEncoder()


# In[59]:


X['image_id']= le.fit_transform(X['image_id'])


# In[60]:


X['image_id'].unique()


# In[61]:


y['image_id']= le.fit_transform(y['image_id'])


# In[62]:


y['image_id'].unique()


# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# # Confusion Matrix graph and final Classfication report 

# In[65]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
SVC(random_state=0)
predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


# In[66]:


from sklearn import metrics


# In[67]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[68]:


Precision = metrics.precision_score(y_test, predictions)


# In[69]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[70]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[71]:


F1_score = metrics.f1_score(y_test, predictions)


# In[72]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:





# In[73]:


BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=20


# In[75]:


import tensorflow as tf


# In[76]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "G:\capstone\project\Apple_Trees/plant_images/apple",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[77]:


class_names = dataset.class_names
class_names


# In[78]:


for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())


# In[79]:


plt.figure(figsize=(10, 10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")


# In[90]:


import numpy as np
for images_batch, labels_batch in dataset.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[1])])


# In[89]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[82]:


plt.figure(figsize=(15, 15))
for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")


# In[ ]:





# In[ ]:





# In[ ]:




