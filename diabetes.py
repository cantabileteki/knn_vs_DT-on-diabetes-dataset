import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

diabetes_data=pd.read_csv("E:\TAI\diabetes.csv ",encoding='utf-8')
#display dataset shape,line9 and line11 referenced [1]
print(diabetes_data.shape)
pd.set_option('display.max_columns',10)  #display all columns,referenced [2]
print(diabetes_data)
print("\n")

print(diabetes_data.head(8))  #print first 8 lines in dataset
diabetes_data.info()    #show dataset info,referenced [3]
#show outcome diagram,referenced [4]
sns.countplot(diabetes_data['Outcome'],label="Count")
plt.show()
#show outcome numerical distrubution,referenced [4]
print(diabetes_data .groupby('Outcome').size())

#show bar chart for each attribute,referenced [5]
diabetes_data.hist(figsize=(8,7))
plt.show()

#observe correlation between variables,referenced [6]
print(diabetes_data.corr())
#pregnancies, glucose, BMI, age have most corr
print('\n')

#process data:
#check whether there are null values,referenced [7]
print(diabetes_data.isnull().sum())
#no missing data
print('\n')

#Judge extreme exceptions of pregnancies?
#3 sigma checking principle,referenced [8]
pre_mean=diabetes_data['Pregnancies'].mean()
pre_td = diabetes_data['Pregnancies'].std()
pre_max =pre_mean+3*pre_td
pre_min = pre_mean-3*pre_td

print("Normal range of pregnancies：",pre_max,pre_min)
print("whether excep> max：",any(diabetes_data ['Pregnancies']>pre_max ))
print("whether excep< min：",any(diabetes_data ['Pregnancies']<pre_min))
#find extreme exception of pregnancies: (finding method referenced [8])
print(diabetes_data.loc[(diabetes_data["Pregnancies"]>pre_max)|(diabetes_data["Pregnancies"]<pre_min),['Pregnancies','Outcome']])
#not processing pregnancies' exceptions
print('\n')

#judge extreme exceptions of glucose?
glu_mean=diabetes_data['Glucose'].mean()
glu_td = diabetes_data['Glucose'].std()
glu_max =glu_mean+3*glu_td
glu_min = glu_mean-3*glu_td

print("Normal range of glucose：",glu_max,glu_min)
print("whether excep> max：",any(diabetes_data ['Glucose']>glu_max ))
print("whether excep< min：",any(diabetes_data ['Glucose']<glu_min))
#find extreme exception of glucose:
print(diabetes_data.loc[(diabetes_data["Glucose"]>glu_max)|(diabetes_data["Glucose"]<glu_min),['Glucose','Outcome']])
#delete points with glucose==0 later
print('\n')

#judge extreme exceptions of blood pressure?
bloo_mean=diabetes_data['BloodPressure'].mean()
bloo_td = diabetes_data['BloodPressure'].std()
bloo_max =bloo_mean+3*bloo_td
bloo_min = bloo_mean-3*bloo_td

print("Normal range of blood pressure：",bloo_max,bloo_min)
print("whether excep> max：",any(diabetes_data ['BloodPressure']>bloo_max ))
print("whether excep< min：",any(diabetes_data ['BloodPressure']<bloo_min))

#find extreme exceptions of blood pressure:
print(diabetes_data.loc[(diabetes_data["BloodPressure"]>bloo_max)|(diabetes_data["BloodPressure"]<bloo_min),['BloodPressure','Outcome']])
#delete points with blood pressure==0 later
print('\n')

#judge extreme exceptions of skin thickness?
skin_mean=diabetes_data['SkinThickness'].mean()
skin_td = diabetes_data['SkinThickness'].std()
skin_max =skin_mean+3*skin_td
skin_min = skin_mean-3*skin_td

print("Normal range of skin thickness：",skin_max,skin_min)
print("whether excep> max：",any(diabetes_data ['SkinThickness']>skin_max ))
print("whether excep< min：",any(diabetes_data ['SkinThickness']<skin_min))

#find extreme exception of skin:
print(diabetes_data.loc[(diabetes_data["SkinThickness"]>skin_max)|(diabetes_data["SkinThickness"]<skin_min),['SkinThickness','Outcome']])
#just one extreme exception(skin thickness==99), not processing it
print('\n')

#judge extreme exceptions of insulin?
insu_mean=diabetes_data['Insulin'].mean()
insu_td = diabetes_data['Insulin'].std()
insu_max =insu_mean+3*insu_td
insu_min = insu_mean-3*insu_td

print("Normal range of insulin：",insu_max,insu_min)
print("whether excep> max：",any(diabetes_data ['Insulin']>insu_max ))
print("whether excep< min：",any(diabetes_data ['Insulin']<insu_min))
#find extreme exception of insulin:
print(diabetes_data.loc[(diabetes_data["Insulin"]>insu_max)|(diabetes_data["Insulin"]<insu_min),['Insulin','Outcome']])
#Just leave
print('\n')

#Judge extreme exceptions of BMI?
BMI_mean=diabetes_data['BMI'].mean()
BMI_td = diabetes_data['BMI'].std()
BMI_max =BMI_mean+3*BMI_td
BMI_min = BMI_mean-3*BMI_td

print("Normal range of BMI：",BMI_max,BMI_min)
print("whether excep> max：",any(diabetes_data ['BMI']>BMI_max ))
print("whether excep< min：",any(diabetes_data ['BMI']<BMI_min))
#Find extreme exception of BMI:
print(diabetes_data.loc[(diabetes_data["BMI"]>BMI_max)|(diabetes_data["BMI"]<BMI_min),['BMI','Outcome']])
# Normal BMI range:18-32, 0 :extrmely abnormal( should be deleted); too high: overfat
print('\n')

#Judge extreme exceptions of DiabetesPedigree Funtion?
dfun_mean=diabetes_data['DiabetesPedigreeFunction'].mean()
dfun_td = diabetes_data['DiabetesPedigreeFunction'].std()
dfun_max =dfun_mean+3*dfun_td
dfun_min = dfun_mean-3*dfun_td

print("Normal range of diabetes pedigree function：",dfun_max,dfun_min)
print("whether excep> max：",any(diabetes_data ['DiabetesPedigreeFunction']>dfun_max ))
print("whether excep< min：",any(diabetes_data ['DiabetesPedigreeFunction']<dfun_min))
#find extreme exception of DiabetesPedigree function:
print(diabetes_data.loc[(diabetes_data["DiabetesPedigreeFunction"]>dfun_max)|(diabetes_data["DiabetesPedigreeFunction"]<dfun_min),['DiabetesPedigreeFunction','Outcome']])
#just leave
print('\n')

#judge extreme exceptions of age?
age_mean=diabetes_data['Age'].mean()
age_td = diabetes_data['Age'].std()
age_max =age_mean+3*age_td
age_min = age_mean-3*age_td

print("Normal range of age：",age_max,age_min)
print("whether excep> max：",any(diabetes_data ['Age']>age_max ))
print("whether excep< min：",any(diabetes_data ['Age']<age_min))
#find extreme exception of age:
print(diabetes_data.loc[(diabetes_data["Age"]>age_max)|(diabetes_data["Age"]<age_min),['Age','Outcome']])
#just leave
print('\n')

#copy the diabetes dateset into a new dataframe, then I process dataset on the new one
diabetes_cp = diabetes_data.copy(deep=True )  #referenced [9]

#delete extreme exceptions, referenced [10]
diabetes_cp  = diabetes_cp [~diabetes_cp ['Glucose'].isin([0])] #delete points(glucose==0)
diabetes_cp  = diabetes_cp [~diabetes_cp ['BloodPressure'].isin([0])] #delete points(blood pressure==0)
diabetes_cp = diabetes_cp [~diabetes_cp ['BMI'].isin([0])]  #delete points(bmi==0)
#put the processed dataframe into new file
pd.DataFrame(diabetes_cp ).to_csv ("E:\TAI\diabetes_datacopy.csv")
#not deleting columns lowly correlated with outcome


# train dataset and evaluate training effect
#use KNN algorithm, referenced [4]

from sklearn.neighbors import KNeighborsClassifier

#cross validation

from sklearn.model_selection import cross_val_score   #referenced [11]

#k: number of knn neighbors change in [1,20]
k=range(1,21)
X=diabetes_cp.iloc[:, 0:8] #independent data columns
Y=diabetes_cp['Outcome']  #dependent data columns
knncr=[]  #used to record knn cross_validation score at each neighbors' number in [1,20]
for i in k:
    knn_c = KNeighborsClassifier(n_neighbors=i)    #fit knn model at each neighbors' number
    kscores=cross_val_score(knn_c,X,Y,cv=3,scoring='accuracy')  #get each model's cross_validation score(3 folded)
    knncr.append(kscores.mean())   #solve each mean value of 3 folded cross_validation score, then record them into array
#draw plot
plt.plot(k, knncr, label="cross accuracies")
plt.ylabel("cross_accuracy knn")
plt.xlabel("neighbors")
plt.legend()
plt.show()
#score achieves highest at neighbors==11, as observed
print("The knn's score(cross validation):", knncr[10],"achieved at 11")


#cross_val_predict is used to predict the classes rather than accuracy
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
#because knn achieves best at neighbors=11, so choose to analyze confusion matrix at this, referenced [12]
knn_c_11 = KNeighborsClassifier(n_neighbors=11)
kpre=cross_val_predict(knn_c_11,X,Y,cv=3)          #predict classes using cross validation
prematrix=confusion_matrix(Y,kpre)             #compute confusion matrix
print(prematrix)
sns.set()
f,ax=plt.subplots()
sns.heatmap(prematrix,annot=True,fmt="g",ax=ax)
ax.set_title("confusion matrix(0: non diabetes, 1:diabetes)")
ax.set_xlabel("predict")
ax.set_ylabel("true")
plt.show()          #draw the confusion matrix, referenced [13]


#decision tree, referenced [14]
from sklearn import tree
from mpl_toolkits.mplot3d import Axes3D
#r:random states change in [10,80]
r=range(10,81)

#explore the most approriate max_depth within [3,6] of the decision tree
#max_depth=3, cross validation
#draw the plot of decision tree's cross validation scores within random states[10,80]
dsco2=[]
for i in r:
    clfcr = tree.DecisionTreeClassifier(criterion="entropy", random_state=i, splitter="random", max_depth=3)
    dsco2_each=cross_val_score(clfcr ,X,Y,cv=3,scoring='accuracy')  #3 folded
    dsco2.append(dsco2_each .mean())

plt.plot(r, dsco2 , label="cross accuracies_3")
plt.ylabel("tree cross_accuracy_3")
plt.xlabel("random states")
plt.legend()
plt.show()
#solve the max value of cross_validation scores under max_depth==3
maxcr=max(dsco2 )
inx_cr=dsco2.index(maxcr )
print("The Decision Tree's score(cross validation):",maxcr,"(max_depth:3 random states:",inx_cr+10,")" )

#max_depth=4 (steps same as above)
dsco2_4=[]
for i in r:
    clfcr4 = tree.DecisionTreeClassifier(criterion="entropy", random_state=i, splitter="random", max_depth=4)
    dsco2_each_4=cross_val_score(clfcr4 ,X,Y,cv=3,scoring='accuracy')
    dsco2_4.append(dsco2_each_4 .mean())

plt.plot(r, dsco2_4  , label="cross accuracies_4")
plt.ylabel("tree cross_accuracy_4")
plt.xlabel("random states")
plt.legend()
plt.show()
maxcr_4=max(dsco2_4 )
inx_cr_4=dsco2_4.index(maxcr_4 )
print("The Decision Tree's score(cross validation):",maxcr_4,"(max_depth:4 random states:",inx_cr_4+10,")" )

#max_depth==5
dsco2_5=[]
for i in r:
    clfcr5 = tree.DecisionTreeClassifier(criterion="entropy", random_state=i, splitter="random", max_depth=5)
    dsco2_each_5=cross_val_score(clfcr5 ,X,Y,cv=3,scoring='accuracy')
    dsco2_5.append(dsco2_each_5 .mean())

plt.plot(r, dsco2_5  , label="cross accuracies_5")
plt.ylabel("tree cross_accuracy_5")
plt.xlabel("random states")
plt.legend()
plt.show()
maxcr_5=max(dsco2_5 )
inx_cr_5=dsco2_5.index(maxcr_5 )
print("The Decision Tree's score(cross validation):",maxcr_5,"(max_depth:5 random states:",inx_cr_5+10,")")

#max_depth==6
dsco2_6=[]
for i in r:
    clfcr6 = tree.DecisionTreeClassifier(criterion="entropy", random_state=i, splitter="random", max_depth=6)
    dsco2_each_6=cross_val_score(clfcr6 ,X,Y,cv=3,scoring='accuracy')
    dsco2_6.append(dsco2_each_6 .mean())

plt.plot(r, dsco2_6  , label="cross accuracies_6")
plt.ylabel("tree cross_accuracy_6")
plt.xlabel("random states")
plt.legend()
plt.show()
maxcr_6=max(dsco2_6 )
inx_cr_6=dsco2_6.index(maxcr_6 )
print("The Decision Tree's score(cross validation):",maxcr_6,"(max_depth:6 random states:",inx_cr_6+10,")")


#draw confusion matrix at best DT model(max_depth==4,random states==47)
clfcr4_47 = tree.DecisionTreeClassifier(criterion="entropy", random_state=47, splitter="random", max_depth=4)    #refit model
dtpre=cross_val_predict(clfcr4_47,X,Y,cv=3)          #predict classes using cross validation
dtprematrix=confusion_matrix(Y,dtpre)             #compute confusion matrix
print(dtprematrix)
f,ax=plt.subplots()
sns.heatmap(dtprematrix,annot=True,fmt="g",ax=ax)
ax.set_title("confusion matrix(0: non diabetes, 1:diabetes)")
ax.set_xlabel("predict")
ax.set_ylabel("true")
plt.show()                                     #draw the confusion matrix

#print features' importance at best DT model(max_depth==4,random states==47), referenced [15],[16]
feature_name=['Pregnancy','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigree Function','Age']
clfcr4_47.fit(X,Y)     #I need to refit the model. because cross_val_predict only returns the predicted classes
print(clfcr4_47.feature_importances_)

#draw features' importance, referenced [17]
plt.figure(figsize=(9,7))
n_features = 8
plt.bar(feature_name, clfcr4_47.feature_importances_)
plt.ylabel("feature importance")
plt.xlabel("Feature")
x_t = list(range(len(feature_name)))
plt.show()



