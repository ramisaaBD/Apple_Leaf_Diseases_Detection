#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import os
from re import search
import shutil
import natsort
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


# In[6]:


DIR=r'D:\Python37\Projects\Foliar diseases in apple trees\images\Original Dataset'


# In[7]:


DIR=r'G:\capstone\project\Apple_Trees\plant_img/images'


# In[8]:


train=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\train.csv")
test=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\test.csv")


# In[9]:


train.head()


# In[10]:


test.head()


# In[11]:


image1=Image.open(r'G:\capstone\project\Apple_Trees\plant_img\images/Test_1.jpg')
plt.imshow(image1)
plt.show()


# # Healthy

# In[17]:


image1=Image.open(r'G:\capstone\project\Apple_Trees\plant_images\apple\apple_healthy/Train_646.jpg')
plt.imshow(image1)
plt.show()


# In[18]:


import os

count = 0
for root_dir, cur_dir, files in os.walk(r'G:\capstone\project\Apple_Trees\plant_images\apple\apple_healthy'):
    count += len(files)
print('file count:', count)


# # Rust

# In[19]:


image1=Image.open(r'G:\capstone\project\Apple_Trees\plant_images\apple\apple_rust/Train_403.jpg')
plt.imshow(image1)
plt.show()


# In[20]:


import os

count = 0
for root_dir, cur_dir, files in os.walk(r'G:\capstone\project\Apple_Trees\plant_images\apple\apple_rust'):
    count += len(files)
print('file count:', count)


# # Scab

# In[21]:


image1=Image.open(r'G:\capstone\project\Apple_Trees\plant_images\apple\apple_scab/Train_29.jpg')
plt.imshow(image1)
plt.show()


# In[22]:


import os

count = 0
for root_dir, cur_dir, files in os.walk(r'G:\capstone\project\Apple_Trees\plant_images\apple\apple_scab'):
    count += len(files)
print('file count:', count)


# # Multiple

# In[23]:


image1=Image.open(r'G:\capstone\project\Apple_Trees\plant_images\apple\apple_multiple/Train_180.jpg')
plt.imshow(image1)
plt.show()


# In[24]:


import os

count = 0
for root_dir, cur_dir, files in os.walk(r'G:\capstone\project\Apple_Trees\plant_images\apple\apple_multiple'):
    count += len(files)
print('file count:', count)


# In[25]:


import matplotlib.pyplot as plt


# In[26]:


a = ['Multiple','Healthy','Scab','Rust']


# In[27]:


b=['91','516','592','622']


# In[28]:


plt.plot(a,b)
plt.ylabel("Data")
plt.xlabel("Classification")
plt.title("Data Set graph")
plt.show()


# In[29]:


plt.bar(a,b)
plt.ylabel("Data")
plt.xlabel("Classification")
plt.title("Data Set graph")
plt.show()


# In[30]:


plt.scatter(a,b)
plt.ylabel("Data")
plt.xlabel("Classification")
plt.title("Data Set graph")
plt.show()


# In[ ]:





# In[ ]:





# # # Prepare the Training Data

# In[31]:


class_names=train.loc[:,'healthy':].columns
print(class_names)

number=0
train['label']=0
for i in class_names:
    train['label']=train['label'] + train[i] * number
    number=number+1

train.head()

DIR

natsort.natsorted(os.listdir(DIR))

def get_label_img(img):
    if search("Train",img):
        img=img.split('.')[0]
        label=train.loc[train['image_id']==img]['label']
        return label

def create_train_data():
    images=natsort.natsorted(os.listdir(DIR))
    for img in tqdm(images):
        label=get_label_img(img)
        path=os.path.join(DIR,img)
        
        if search("Train",img):
            if (img.split("_")[1].split(".")[0]) and label.item()==0:
                shutil.copy(path,r'G:\capstone\project\Apple_Trees\plant_images\apple/apple_healthy')
            
            elif(img.split("_")[1].split(".")[0]) and label.item()==1:
                shutil.copy(path,r'G:\capstone\project\Apple_Trees\plant_images\apple/apple_multiple')
                
            elif(img.split("_")[1].split(".")[0]) and label.item()==2:
                shutil.copy(path,r'G:\capstone\project\Apple_Trees\plant_images\apple/apple_rust')
                
            elif(img.split("_")[1].split(".")[0]) and label.item()==3:
                shutil.copy(path,r'G:\capstone\project\Apple_Trees\plant_images\apple/apple_scab')
                
        elif search("Test",img):
            shutil.copy(path,r'G:\capstone\project\Apple_Trees\split_class_img/test')



train_dir=create_train_data()


# In[32]:


Train_DIR=r'G:\capstone\project\Apple_Trees\split_class_img/train'
Categories=['healthy','multiple_disease','rust','scab']

for j in Categories:
    path=os.path.join(Train_DIR,j)
    for img in os.listdir(path):
        old_image=cv2.imread(os.path.join(path,img),cv2.COLOR_BGR2RGB)
        plt.imshow(old_image)
        plt.show()
        break
    break

IMG_SIZE=224
new_image=cv2.resize(old_image,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_image)
plt.show()


# In[33]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Activation,Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy

datagen=ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                validation_split=0.2)


train_datagen=datagen.flow_from_directory(r'G:\capstone\project\Apple_Trees\split_class_img/train',
                                         target_size=(IMG_SIZE,IMG_SIZE),
                                         batch_size=16,
                                         class_mode='categorical',
                                         subset='training')

val_datagen=datagen.flow_from_directory(r'G:\capstone\project\Apple_Trees\split_class_img/train',
                                         target_size=(IMG_SIZE,IMG_SIZE),
                                         batch_size=16,
                                         class_mode='categorical',
                                         subset='validation')

class_names = train_datagen.class_indices
print(class_names)

model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(4,activation='softmax'))


# # # Compile the Model

# In[34]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary()



checkpoint=ModelCheckpoint(r'E:\New folder (2)\New folder\models-20221115T174943Z-001\models/apple2.h5',
                          monitor='val_loss',
                          mode='min',
                          save_best_only=True,
                          verbose=1)
earlystop=EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=10,
                       verbose=1,
                       restore_best_weights=True)

callbacks=[checkpoint,earlystop]

model_history=model.fit_generator(train_datagen,validation_data=val_datagen,
                                 epochs=30,
                                 steps_per_epoch=train_datagen.samples//16,
                                 validation_steps=val_datagen.samples//16,
                                  callbacks=callbacks)



import plotly.graph_objects as go
from plotly.subplots import make_subplots


# # Create figure with secondary y-axis

# In[35]:


fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter( y=model_history.history['val_loss'], name="val_loss"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter( y=model_history.history['loss'], name="loss"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter( y=model_history.history['val_accuracy'], name="val accuracy"),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter( y=model_history.history['accuracy'], name="accuracy"),
    secondary_y=True,
)


# In[36]:


# Add figure title
fig.update_layout(
    title_text="Loss/Accuracy of Foliar diseases in apple trees Model"
)

# Set x-axis title
fig.update_xaxes(title_text="Epoch")

# Set y-axes titles
fig.update_yaxes(title_text="<b>primary</b> Loss", secondary_y=False)
fig.update_yaxes(title_text="<b>secondary</b> Accuracy", secondary_y=True)

fig.show()

acc_train=model_history.history['accuracy']
acc_val=model_history.history['val_accuracy']
epochs=range(1,31)
plt.plot(epochs,acc_train,'g',label='Training Accuracy')
plt.plot(epochs,acc_val,'b',label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

loss_train=model_history.history['loss']
loss_val=model_history.history['val_loss']
epochs=range(1,31)
plt.plot(epochs,loss_train,'g',label='Training Loss')
plt.plot(epochs,loss_val,'b',label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[37]:


import keras
from matplotlib import pyplot as plt
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.show()


# In[40]:



from tensorflow import keras
model = keras.models.load_model(r'G:\capstone\project\Apple_Trees\models\apple2.h5')

test_image=r'G:\capstone\project\Apple_Trees\split_class_img\test/Test_1203.jpg'
image_result=Image.open(test_image)

from tensorflow.keras.preprocessing import image
test_image=image.load_img(test_image,target_size=(224,224))
test_image=image.img_to_array(test_image)
test_image=test_image/255
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
print(np.argmax(result))
Categories=['healthy','multiple_disease','rust','scab']
image_result=plt.imshow(image_result)
plt.title(Categories[np.argmax(result)])
plt.show()


# In[41]:


help(model)


# In[42]:


from keras import metrics


# In[43]:


model.compile(loss='mean_squared_error', optimizer='sgd',
              metrics=[metrics.mae,
                       metrics.categorical_accuracy])


# In[44]:


keras.metrics.categorical_accuracy(test_image, result)


# In[45]:


import seaborn as sns


# In[46]:


from sklearn.preprocessing import LabelEncoder


# In[47]:


le =LabelEncoder()


# In[48]:


train['image_id']= le.fit_transform(train['image_id'])


# In[49]:


train['image_id'].unique()


# In[50]:


sns.heatmap(train, vmin=50, vmax=100)


# In[51]:


sns.heatmap(result, annot=True,xticklabels=test_image, yticklabels=test_image)


# In[52]:


y_pred_class = model.compile(test_image) 
y_pred = model.predict(test_image)               
y_test_class = np.argmax(test_labels, axis=1)     

print(classification_report(y_test_class, y_pred_class))


# In[53]:


model.compile(metrics=[1,224,224,3])


# In[54]:


def recall(test_image, result):
    y_true = K.ones_like(test_image)
    true_positives = K.sum(K.round(K.clip(test_image * result, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


# In[55]:


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


# In[56]:


model.compile(metrics=['accuracy', f1_score, precision, recall])


# In[57]:


test_image.shape


# In[58]:


# reduce to 1d array
result = result[:, 0]
test_image = test_image[:, 0]


# In[59]:


model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.AUC()])


# In[60]:


model.compile('sgd', loss='mse',
               metrics=[tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall()])


# In[61]:


print(recall)


# In[62]:


from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[63]:


train=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\train.csv")
test=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\test.csv")


# In[64]:


le =LabelEncoder()


# In[65]:


train['image_id']= le.fit_transform(train['image_id'])


# In[66]:


train['image_id'].unique()


# In[67]:


le =LabelEncoder()


# In[68]:


test['image_id']= le.fit_transform(test['image_id'])


# In[69]:


test['image_id'].unique()


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42)


# In[71]:


logmodel = tree.DecisionTreeClassifier()


# In[72]:


logmodel.fit(X_train, y_train)


# In[73]:


predictions = logmodel.predict(X_test)


# In[74]:


classification_report(y_test,predictions)


# In[76]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


# In[77]:


X=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\train.csv")
y=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\test.csv")


# In[78]:


from sklearn.preprocessing import LabelEncoder


# In[79]:


le =LabelEncoder()


# In[80]:


X['image_id']= le.fit_transform(X['image_id'])


# In[81]:


X['image_id'].unique()


# In[82]:


y['image_id']= le.fit_transform(y['image_id'])


# In[83]:


y['image_id'].unique()


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[85]:


classifier_tree = DecisionTreeClassifier()
y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[88]:


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


# In[89]:


from sklearn import metrics


# In[90]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[91]:


Precision = metrics.precision_score(y_test, predictions)


# In[92]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[93]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[94]:


F1_score = metrics.f1_score(y_test, predictions)


# In[95]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:





# In[ ]:




