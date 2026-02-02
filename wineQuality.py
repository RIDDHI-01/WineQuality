import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
wine_df = pd.read_csv("Lab 22 winequality-red.csv")
wine_df.head()
wine_df.tail()
wine_df.sample(7)
print(wine_df.columns)
print(wine_df.shape)
wine_df.info()
sns.set(style="whitegrid")
print(wine_df['quality'].value_counts())
fig = plt.figure(figsize = (10,6))
sns.countplot(x='quality',data=wine_df, palette='pastel')
plt.figure(figsize = (10,8))
sns.heatmap(wine_df.corr(),annot=True, cmap= 'PuBuGn')
color = sns.color_palette("pastel")

fig, ax1 = plt.subplots(3,4, figsize=(20,30))
k = 0
columns = list(wine_df.columns)
for i in range(3):
    for j in range(4):
            sns.distplot(wine_df[columns[k]], ax = ax1[i][j], color = 'red')
            k += 1
plt.show()
wine_df.corr()['quality'].sort_values(ascending=False)
X = wine_df.drop(columns=['quality'])
y = wine_df['quality']
y.value_counts()
from imblearn.over_sampling import SMOTE
oversample = SMOTE(k_neighbors=4)
# transform the dataset
X, y = oversample.fit_resample(X.fillna(0), y)
y.value_counts()
from sklearn.model_selection import cross_val_score, train_test_split

def classify(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # train the model
    model.fit(x_train, y_train)
    return model.score(x_test, y_test) * 100
    #print("Accuracy:", model.score(x_test, y_test) * 100)
    
#     # cross-validation
#     score = cross_val_score(model, X, y, cv=5)
#     print("CV Score:", np.mean(score)*100)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
LinearReg_acc=classify(model, X, y)
LinearReg_acc
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
DecTree_acc=classify(model, X, y)
DecTree_acc
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
RanFor_acc=classify(model, X, y)
RanFor_acc
Accuracy = [LinearReg_acc, DecTree_acc,RanFor_acc,SVM_acc]
models = ['LogisticRegression', 'DecisionTreeClassifier' , 'RandomForestClassifier', 'Support Vector Machine']
sns.barplot(x=Accuracy, y=models, color="g")
plt.xlabel('Accuracy in %')
plt.title('Accuracy')
plt.show()
from sklearn.ensemble import RandomForestClassifier
RanFor_model = RandomForestClassifier()
RanFor_acc=classify(RanFor_model, X, y)
RanFor_acc
import pickle
# save the model to disk
filename = 'finalRF_model.sav'
pickle.dump(RanFor_model, open(filename, 'wb'))
# Load the Model back from file
with open(filename, 'rb') as file:  
    RF_Model = pickle.load(file)

RF_Model