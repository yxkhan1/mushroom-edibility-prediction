import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('mushrooms.csv')

##### Deleting the output colum
y=df['class']
y=[1 if i=='e' else 0 for i in y]
#Delet class from data
del df['class']

#Ordinal Econding method
#Most of the catergorical features are ranging between a-z let convert the to numerical values between 0-26 using Ordinal Encoding
alpha={'?':0,'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,
       'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z':26}
df1=pd.DataFrame()
for i in df:
    df1[i]=df[i].map(alpha)

#Feature importance
from sklearn.ensemble import ExtraTreesClassifier
m=ExtraTreesClassifier()
m.fit(df1,y)
print(m.feature_importances_)
#plot the 5 highest models
feat_importances=pd.Series(m.feature_importances_,index=df.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

# #New Data with 5 importan features
d1=pd.DataFrame()
d1['odor']=df1['odor']
d1['gill-size']=df1['gill-size']
d1['bruises']=df1['bruises']
d1['gill-color']=df1['gill-color']
d1['gill-spacing']=df1['gill-spacing']


#### Split Data into Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(d1,y,test_size=0.20,random_state=42)

## Model Building
#Passing the Parameters
from sklearn.neighbors import KNeighborsClassifier
neigh1=KNeighborsClassifier(n_neighbors=10)
#Training the Model
knn_model1=neigh1.fit(X_train, y_train)
#Prediction
knn_predict_y1=knn_model1.predict(X_test)
# Evultion Metrics
from sklearn.metrics import *
print(mean_squared_error(knn_predict_y1,y_test))

#Dumping the model into pickel file
import pickle
pickle.dump(knn_model1, open('model.pkl', 'wb'))