#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


train=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\train.csv")
test=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\test.csv")


# # Data Preprocessing

# In[3]:


from sklearn.preprocessing import LabelEncoder


# In[4]:


le =LabelEncoder()


# In[5]:


train['image_id']= le.fit_transform(train['image_id'])


# In[6]:


train['image_id'].unique()


# In[7]:


train.head()


# In[8]:


le =LabelEncoder()


# In[9]:


test['image_id']= le.fit_transform(test['image_id'])


# In[10]:


test['image_id'].unique()


# In[11]:


test.head()


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42)


# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)


# In[15]:


from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)

test_1 = X_test.iloc[1]


# In[128]:


X_train.shape


# # Neural Network

# In[16]:


from sklearn.neural_network import MLPClassifier


# In[17]:


anna = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[18]:


anna.fit(X_train,y_train)


# In[19]:


y_train


# In[20]:


y_test


# In[21]:


pred=anna.predict(X_test)
pred


# # Classification report of Neural Network

# In[22]:


from sklearn.metrics import classification_report


# In[23]:


classification_report(y_test,pred)


# In[24]:


classifier_tree = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[25]:


y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[26]:


print(classification_report(y_test, y_predict))


# # Lime

# In[27]:


import lime 
from lime import lime_tabular

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    mode='classification',
    verbose=True
)


lime_exp = lime_explainer.explain_instance(
    data_row=test_1,
    predict_fn=anna.predict_proba,
    num_features=4,top_labels=1
)
lime_exp.show_in_notebook(show_table=True)


# In[28]:


lime_explainer = lime_tabular.LimeTabularExplainer(np.array(train),
    feature_names=train.columns,verbose=True,
    
    mode='classification')


# In[29]:


lime_exp = lime_explainer.explain_instance(
    X_test.iloc[0],
    anna.predict_proba
)
lime_exp.show_in_notebook(show_table=True)


# In[ ]:





# In[30]:


lime_exp.predict_proba


# In[31]:


import numpy as np
import pandas as pd


# In[32]:


wine = pd.read_csv(r'G:\capstone\project\Apple_Trees\plant_img/sample_submission.csv')
wine.head()


# # Data Preprocessing 

# In[33]:


from sklearn.preprocessing import LabelEncoder


# In[34]:


le =LabelEncoder()


# In[35]:


wine['image_id']= le.fit_transform(wine['image_id'])


# In[36]:


wine['image_id'].unique()


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X = wine.drop('image_id', axis=1)
y = wine['image_id']


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[40]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)


# In[41]:


from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)

test_1 = X_test.iloc[1]


# # Neural Network 

# In[42]:


from sklearn.neural_network import MLPClassifier


# In[43]:


anna = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[44]:


anna.fit(X_train,y_train)


# In[45]:


pred=anna.predict(X_test)
pred


# # Classfication report of Neural Network

# In[46]:


from sklearn.metrics import classification_report


# In[47]:


classification_report(y_test,pred)


# In[48]:


classifier_tree = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[49]:


y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[50]:


print(classification_report(y_test, y_predict))


# # Confusion matrix of Neural Network

# In[51]:


from sklearn.metrics import confusion_matrix


# In[52]:


confusion_matrix(y_test,pred)


# # Support Vector Machine

# In[53]:


from sklearn import svm


# In[54]:


logmodel = svm.SVC()


# In[55]:


logmodel.fit(X_train, y_train)


# In[56]:


predictions = logmodel.predict(X_test)


# # Classification report of of Support Vector Machine

# In[57]:


from sklearn.metrics import classification_report


# In[58]:


classification_report(y_test,predictions)


# In[59]:


classifier_tree = svm.SVC()


# In[60]:


y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[61]:


print(classification_report(y_test, y_predict))


# # Confuision Matrix of Support vector Machine

# In[62]:


from sklearn.metrics import confusion_matrix


# In[63]:


confusion_matrix(y_test,predictions)


# In[ ]:





# # Decision Tree

# In[64]:


from sklearn import tree


# In[65]:


logmodel = tree.DecisionTreeClassifier()


# In[66]:


logmodel.fit(X_train, y_train)


# In[67]:


predictions = logmodel.predict(X_test)


# # Classification report of Decision tree

# In[68]:


from sklearn.metrics import classification_report


# In[69]:


classification_report(y_test,predictions)


# In[70]:


classifier_tree = tree.DecisionTreeClassifier()


# In[71]:


y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[72]:


print(classification_report(y_test, y_predict))


# # Confusion Matrix of Decision Tree

# In[73]:


from sklearn.metrics import confusion_matrix


# In[74]:


confusion_matrix(y_test,predictions)


# # Lime

# In[76]:


import lime 
from lime import lime_tabular

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    mode='classification'
)


lime_exp = lime_explainer.explain_instance(
    data_row=test_1,
    predict_fn=logmodel.predict_proba
)
lime_exp.show_in_notebook(show_table=True)


# In[77]:


lime_exp.predict_proba


# # Shap

# In[78]:


import xgboost
import shap
import pandas as pd


# In[79]:


X=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\train.csv")
y=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\test.csv")


# In[80]:


from sklearn.preprocessing import LabelEncoder


# In[81]:


le =LabelEncoder()


# In[82]:


X['image_id']= le.fit_transform(X['image_id'])


# In[83]:


X['image_id'].unique()


# In[84]:


y['image_id']= le.fit_transform(y['image_id'])


# In[85]:


y['image_id'].unique()


# In[86]:


from sklearn.model_selection import train_test_split


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[88]:


model = xgboost.XGBRegressor().fit(X, y)


# In[89]:


explainer = shap.Explainer(model)
shap_values = explainer(X)


# In[90]:


shap.plots.waterfall(shap_values[0])


# In[91]:


shap.plots.force(shap_values[0])


# In[92]:


shap.plots.beeswarm(shap_values)


# In[93]:


shap.plots.bar(shap_values)


# In[ ]:





# In[94]:


train=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\train.csv")
test=pd.read_csv(r"G:\capstone\project\Apple_Trees\plant_img\test.csv")


# # Support vector machine confusion matrix graph and final classfication report 

# In[95]:


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


# In[96]:


predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


# In[97]:


from sklearn import metrics


# In[98]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[99]:


Precision = metrics.precision_score(y_test, predictions)


# In[100]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[101]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[102]:


F1_score = metrics.f1_score(y_test, predictions)


# In[103]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# # Neural Network confusion matrix graph and final classification report 

# In[104]:


from sklearn.neural_network import MLPClassifier


# In[105]:


anna = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[106]:


anna.fit(X_train,y_train)


# In[107]:


pred=anna.predict(X_test)


# In[108]:


predictions = anna.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


# In[109]:


from sklearn import metrics


# In[110]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[111]:


Precision = metrics.precision_score(y_test, predictions)


# In[112]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[113]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[114]:


F1_score = metrics.f1_score(y_test, predictions)


# In[115]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# # Decision tree Confusion matrix graph and final classfication report

# In[116]:


from sklearn import tree


# In[117]:


logmodel = tree.DecisionTreeClassifier()


# In[118]:


logmodel.fit(X_train, y_train)


# In[119]:


logmodel.fit(X_train, y_train)


# In[120]:


predictions = logmodel.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


# In[121]:


from sklearn import metrics


# In[122]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[123]:



Precision = metrics.precision_score(y_test, predictions)


# In[124]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[125]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[126]:


F1_score = metrics.f1_score(y_test, predictions)


# In[127]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:





# In[ ]:




