import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import time
from PIL import Image
# from keras.models import load_model
from sklearn.utils import check_matplotlib_support
import streamlit as st
# from keras.preprocessing import image
from tempfile import NamedTemporaryFile
# # from keras.preprocessing.image import load_img
# import keras
# from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
# from keras.layers import Dense, Flatten, Dropout
# from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
import cv2

F1_knn=0
jk=0
F1_dt=0
jd=0
F1_svm=0
js=0
F1_l=0
jl=0
lllr=0
st.title("Loan Status Classifier")
DATA_URL = ('loan_train.csv')
st.markdown("This application is a streamlit dashboard")

@st.cache(allow_output_mutation=True)
@st.cache(persist=True)
@st.cache(suppress_st_warning=True)
def load_train_data(nrows):
    train_data = pd.read_csv(DATA_URL, nrows=nrows)
    # data.dropna(subset=['manufact'], inplace=True)
    return train_data

def load_test_data(nrows):
    test_data = pd.read_csv('loan_test.csv', nrows=nrows)
    # data.dropna(subset=['manufact'], inplace=True)
    return test_data


if st.checkbox("Show Train data", False):
    data_size = st.slider("Rows of Data to be displayed ",100,346,10)
    train_data = load_train_data(data_size)
    st.header('Train Data')
    st.write(train_data)



data = pd.read_csv('loan_train.csv')

# Convert to date time object
data['due_date'] = pd.to_datetime(data['due_date'])
data['effective_date'] = pd.to_datetime(data['effective_date'])
print(data.head())

# Data Visualization and preprocessing
print(data['loan_status'].value_counts())

# Data Visualization
import seaborn as sns

bins = np.linspace(data.Principal.min(), data.Principal.max(), 10)
g = sns.FacetGrid(data, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

bins = np.linspace(data.age.min(), data.age.max(), 10)
g = sns.FacetGrid(data, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

# Preprocessing: Feature Selection & Extraction
data['dayofweek'] = data['effective_date'].dt.dayofweek
bins = np.linspace(data.dayofweek.min(), data.dayofweek.max(), 10)
g = sns.FacetGrid(data, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
data['weekend'] = data['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
print(data.head())

print(data.groupby(['Gender'])['loan_status'].value_counts(normalize=True))
data['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
print(data.head())

data[['Principal','terms','age','Gender','education']].head()

Feature = data[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(data['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

# Feature Selection
X = Feature
y = data['loan_status'].values

# Normalization
X= preprocessing.StandardScaler().fit(X).transform(X)

# ________________________________________________________________________



test_df = pd.read_csv('loan_test.csv')
# Preprocessing: Feature Selection & Extraction
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])


test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek

test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)


test_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

test_df[['Principal','terms','age','Gender','education']].head()

Feature = test_df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(test_df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

test_X = Feature
test_y = test_df['loan_status'].values
test_X = preprocessing.StandardScaler().fit(test_X).transform(test_X)

# st.write(test_X[:10])

# ________________________________________________________________________
def show_evaluation_metrics(F1_knn,jk,F1_dt,jd,F1_svm,js,F1_l,jl,lllr):
    Evaluation_metrics = [ ["KNN", F1_knn, jk, None],
        ["Decision Tree", F1_dt, jd, None],
        ["SVM", F1_svm, js, None],
        ["LogisticRegression",F1_l, jl, lllr]]

    df = pd.DataFrame(Evaluation_metrics, columns = ['Algorithm','Jaccard','F1-score','LogLoss'])

    return df
# ________________________________________________________________________

select = st.selectbox("Select the Classification Algorithm",("K Nearest Neighbor","Decision Tree","Support Vector Machine","Logistic Regression"))
select_model=select
# ________________________________________________________________________
# KNN MODELING
# percent_test_split = st.slider("Select the splitting percentage of Training data",0.0,1.0,value=0.2)#0.2
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
# max_accuracy = 0
# k_val=0
# for k in range(1,10):
#     #Train Model and Predict  
#     from sklearn.neighbors import KNeighborsClassifier
#     neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
#     # predicting
#     predicted_y = neigh.predict(X_test)
#     #Acuracy Evaluation
#     accuracy_test = metrics.accuracy_score(y_test, predicted_y)
#     #accuracy_train = metrics.accuracy_score(y_train, neigh.predict(X_train))
#     if accuracy_test > max_accuracy:
#         max_accuracy = accuracy_test
#         k_val = k
# print(f"Best Accuracy obtain is {max_accuracy} at {k_val}")


k_value = 7

if select_model == 'K Nearest Neighbor':
    st.write("Best Accuracy is obtained when n_neighbors are 7")
    n = st.slider("Select the k-value",1,10,value =7)
    k_value =n


X_train_k = X_train
y_train_k = y_train
y_test_k = y_test

from sklearn.neighbors import KNeighborsClassifier
k = k_value
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# predicting
predicted_y = neigh.predict(X_test)
pred_y_knn = neigh.predict(test_X)
# st.write(pred_y_knn)
F1_knn=f1_score(test_y,pred_y_knn,average='weighted')
jk = jaccard_score(test_y,pred_y_knn,pos_label='PAIDOFF')  
# ________________________________________________________________________

# Decision Tree

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=39084)#39084
print('Shape of X training set {}'.format(X_train.shape),'&',' Size of Y training set {}'.format(y_train.shape))
print('Shape of X test set {}'.format(X_test.shape),'&',' Size of Y test set {}'.format(y_test.shape))
# Finding max_accuracy
# ''max_acc=0
# index=0
# for i in range(100000):
#     from sklearn.model_selection import train_test_split
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=i)
#     from sklearn.tree import DecisionTreeClassifier
#     import sklearn.tree as tree
#     drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
#     #print(drugTree)
#     drugTree.fit(X_train,y_train)
#     predTree = drugTree.predict(X_test)
#     from sklearn import metrics
#     from matplotlib import pyplot as plt
#     acc=metrics.accuracy_score(y_test,predTree)
#     #print('Decisoin Tree accuracy',metrics.accuracy_score(y_test,predTree))
#     print(i)
#     if max_acc<acc:
#         max_acc=acc
#         index=i
# print(max_acc,index)''
X_train_d = X_train
y_train_d = y_train
y_test_d = y_test
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
print(drugTree)
drugTree.fit(X_train,y_train)
predTree = drugTree.predict(X_test)
pred_y_dt = drugTree.predict(test_X)
F1_dt=f1_score(test_y,pred_y_dt,average='weighted')
jd= jaccard_score(test_y,pred_y_dt,pos_label='PAIDOFF')

# ________________________________________________________________________

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
# Modeling with SVM
from sklearn import svm
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train,y_train)
predict_y = svm_model.predict(X_test)

X_train_s = X_train
y_train_s = y_train
y_test_s = y_test

pred_y_svm =svm_model.predict(test_X)
F1_svm=f1_score(test_y,pred_y_svm,average='weighted')
js=jaccard_score(test_y,pred_y_svm,pos_label='PAIDOFF')
# ________________________________________________________________________
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

X_train_l = X_train
y_train_l =y_train
y_test_l= y_test

from sklearn.linear_model import LogisticRegression

LR_model = LogisticRegression(C=0.05, solver="liblinear").fit(X_train,y_train)

predict_y= LR_model.predict(X_test)

predict_y_prob = LR_model.predict_proba(X_test)
pred_y_LR= LR_model.predict(test_X)
pred_y_prob_LR = LR_model.predict_proba(test_X)
F1_l =f1_score(test_y,pred_y_LR,average='weighted')
jl=jaccard_score(test_y,pred_y_LR,pos_label='PAIDOFF')
from sklearn.metrics import log_loss
lllr=log_loss(test_y, pred_y_prob_LR)
# ________________________________________________________________________


if select == 'K Nearest Neighbor':

    from sklearn import metrics
    st.subheader(f'Accuracy Evalution of {select}:')
    st.write('Train set Accuracy',metrics.accuracy_score(y_train_k, neigh.predict(X_train_k)))
    st.write("Test set Accuracy: ", metrics.accuracy_score(y_test_k, predicted_y))

    ## F1_score
    st.write('F1 score:',F1_knn) 
    ## jaccard index
         
    st.write('Jaccard score:',jk)



if select == 'Decision Tree':
    
    st.subheader(f'Accuracy Evalution of {select}:')
    st.write('Decisoin Tree Train  set accuracy',metrics.accuracy_score(y_train_d,drugTree.predict(X_train_d)))
    st.write('Decisoin Tree Test set accuracy',metrics.accuracy_score(y_test_d,predTree))


    st.write('F1 score:',F1_dt )
    st.write('Jaccard score:',jd)



if select == 'Support Vector Machine':
    st.subheader(f'Accuracy Evalution of {select}:')
    st.write('SVM Test set accuracy',metrics.accuracy_score(y_train_s,svm_model.predict(X_train_d)))
    st.write('SVM Test set accuracy',metrics.accuracy_score(y_test_s,predict_y))
    
    st.write('F1 score:',F1_svm)    
    st.write('Jaccard score:',js)





    # F1_score


    
    print('F1 score:',f1_score(y_test,predict_y,average='weighted') )

    # # Jaccard index
   
    print('Jaccard score of Paidoff:',jaccard_score(y_test_s,predict_y,pos_label='PAIDOFF'))
    print('Jaccard score of Collection:',jaccard_score(y_test_s,predict_y,pos_label='COLLECTION'))



if select == 'Logistic Regression':
    
    # Evaluation
    st.subheader(f'Accuracy Evalution of {select}:')
    st.write('LR Test set accuracy',metrics.accuracy_score(y_train_l,LR_model.predict(X_train_l)))
    st.write('LR Test set accuracy',metrics.accuracy_score(y_test_l,predict_y))


    ## confusion matrix

    from sklearn.metrics import classification_report, confusion_matrix
    import itertools
    def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    print(confusion_matrix(y_test, predict_y, labels=['COLLECTION','PAIDOFF']))
  
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, predict_y, labels=['COLLECTION','PAIDOFF'])
    np.set_printoptions(precision=2)


    # Plot non-normalized confusion matrix
    plt.figure()
    
    fig =plot_confusion_matrix(cnf_matrix, classes=['COLLECTION','PAIDOFF'],normalize= False,  title='Confusion matrix')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)
    print (classification_report(y_test, predict_y))


    st.write('F1 score:',F1_l)
    st.write('Jaccard score:',jl)
    st.write('Log loss:',lllr)

    ## jaccard index
    from sklearn.metrics import jaccard_score
    print(jaccard_score(y_test,predict_y,pos_label='PAIDOFF'))


    # log loss
    from sklearn.metrics import log_loss
    print(log_loss(y_test, predict_y_prob))

data_frame = show_evaluation_metrics(F1_knn,jk,F1_dt,jd,F1_svm,js,F1_l,jl,lllr)

if st.checkbox("Show Evaluation Matrix:", False):
    # st.header('Train Data')
    st.write(data_frame)

if st.checkbox("Show Test data", False):
    
    data_size = st.slider("Rows of Data to be displayed ",10,55,10)
    test_df = load_test_data(data_size)
    st.header('Test Data')
    st.write(test_df)

model = {'K Nearest Neighbor':neigh,'Decision Tree':drugTree,'Support Vector Machine':svm_model,'Logistic Regression':LR_model}
model = model[select_model]


    


col1,col2 = st.columns(2)
with st.sidebar:

    with st.spinner("Predicting..."):
        data_size = st.slider("Rows of Data to predict ",1,54,1)
        test = test_X[:data_size]
        y=test_y[:data_size]
        if st.checkbox("Lets Predict!", False):
            if test is not None:
                result = model.predict(test)
                col1.subheader('Actual')
                col1.write(y)
                col2.subheader('Predicted')
                col2.write(result)
                # st.write(result)
            time.sleep(5)

    st.success("Done!")


