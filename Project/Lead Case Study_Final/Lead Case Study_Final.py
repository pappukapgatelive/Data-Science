#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Lead scoring Case Study
# By Pappu Kapgate & Shivram J


# # **Step 1: Data Understanding**

# ### Step 1.1: Importing and Describeing Data

# In[2]:


# Warnings
import warnings
warnings.filterwarnings('ignore')

#Importing required lib`a
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns


# In[3]:


# Loading the data set
lead_df = pd.read_csv('Leads.csv')
lead_df.head()


# In[4]:


#Exploring data types and column with null values
lead_df.info()


# In[5]:


#Checking the shape for record and collumns counts
lead_df.shape


# In[6]:


#Checking data values in numeric valued columns
lead_df.describe()


# ### Step 1.2: Data Cleaning

# - **1.2.1  Handling Duplicate values**

# In[7]:


# checking the value count of lead Source variable
lead_df['Lead Source'].value_counts()


# ### **Observation:**
# 
# <font color ='green'> Seeing both upcase and lower case values for google</font>
# 
# <font color ='green'> There may be other columns with similar dataset - hence converting all columns to lower case.

# In[8]:


#converting all columns data to lower case due to above observation
lead_df = lead_df.applymap(lambda x:x.lower() if type(x) == str else x) 


# - **1.2.2 Handling Missing Values**

# <font color ="green"> We are seeing select in multiple column values, this could be because the user did not select a value in the UI.
# 
# <font color ="green"> Hence converting it into null

# In[9]:


# replacing the select value with null value
lead_df.replace('select',np.nan, inplace=True)


# - **1.2.3 Handling Single valued column**

# In[10]:


# Identifying the columns having same values in all rows.
temp=lead_df.nunique()
#temp will store columns name having a single value
temp[temp==1]


# ### **Observation:**
# <font color ="green">Dropping all columns identified above which have only 1 value in it.

# In[11]:


# Drop unique valued columns those are present in temp 
lead_df.drop(axis = 1, columns=temp[temp==1].index, inplace=True) 
# Print dropeed column
print("Columns dropped are \n",temp[temp==1].index)


# - **1.2.4 Handling Null values**

# In[12]:


#function to check for list of columns having null_values
def null_value_check(x): # accepting parameter
    temp=100*(lead_df.isna().sum()/lead_df.shape[0]) # calculating the percenteg of null value and storing into temp
    return (temp[temp>x]) # will return only those columns according to the parameter recived by function


# In[13]:


#List of null values greater than 0%
null_value_check(0)


# In[14]:


#List of null values greater than 30%
null_value_check(30)


# ### **Observation:**
# <font color="green"> Apart from specialization we feel all the other columns listed above can be dropped due to a high rate of missing values

# In[15]:


# drope only high null values columns
lead_df = lead_df.drop(['Asymmetrique Profile Index','Asymmetrique Activity Index',
                        'Asymmetrique Activity Score','Asymmetrique Profile Score',
                        'Lead Profile','Tags',
                        'Lead Quality','How did you hear about X Education','City'],axis=1)


# <font color = 'green'> Now checking columns greater than 25% missing values

# In[16]:


# checking null values columns, greater than 25%
col_name = np.array(null_value_check(25).index) # calling a function by passing the paramenter as 25 & storing into col_name 
print(col_name)


# In[17]:


#Fill missing values with 'unknown'

for i in range(len(col_name)):                            # calulating length of col_name list
    lead_df[col_name[i]].fillna('unknown',inplace = True) # replacing null values with the "unknown"


# In[18]:


#checking again for remaining missing values
null_value_check(0)


# In[19]:


# Checking the percent of data loss if we removed the null values rows from the data frame
round(100*(sum(lead_df.isnull().sum(axis=1) > 1)/lead_df.shape[0]),2)


# In[20]:


# Removing rows with have very less null values 
lead_df = lead_df[lead_df.isnull().sum(axis=1) <1]


# In[21]:


# All missing values are treated!
null_value_check(0)


# ### **Observation:**
# 
#     - We conclude that, all the null values are get handaled properly.

# - **1.2.5 Handling Unique Value variable**

#    - Removing Id values since they are unique for everyone, we will be leveraging Lead number to identify unique leads.

# In[22]:


# Removing Id values since they are unique for everyone, we will be leveraging Lead number to identify unique leads.
lead_df.drop('Prospect ID',axis = 1,inplace = True)
# checking shape
lead_df.shape


# - Checking for Data Imbalance in Binary variables

# In[23]:


# Listing out all binary variables and their values counts. cat_variables contain the binary column names.

def cat_variable_identification(var1, var2):
    cat_variables=list(lead_df.select_dtypes(var1).columns) # extracting only those columns, who's data type is specified in function parameter and storing into the "cat_variables"
    for i in range(len(cat_variables)):
        if len((lead_df[cat_variables[i]].value_counts().index)) <= var2: # if the range of index is equals to the two then print only those columns
            print((lead_df[cat_variables[i]].value_counts()))
            print("\n")
            print("-------------------")
            


# In[24]:


cat_variable_identification("object",2)


# ### **Observation:**
# 
# - we observed that the many variables of dataFrame consite the single value, i.e very data immbalance. It is, therfore, we are dropping such columns.The columns are listed below that we are dropping:
# 
# - Do Not Call
# 
# - Search
# 
# - Newspaper Article
# 
# - X Education Forums
# 
# - Newspaper 
# 
# - Digital Advertisement
# 
# - Through Recommendations'

# In[25]:


# Drop the below columns as they have data immbalance.
lead_df.drop(['Do Not Call','Search','Newspaper Article','X Education Forums',
              'Newspaper','Digital Advertisement','Through Recommendations'],axis=1,inplace=True)


# In[26]:


#List the remaining binary variables.
cat_variable_identification('object',2) # passing the paramenters


# - Checking for Data Imbalance in Categorical variables

# In[27]:


# Listing out all category variables and their values counts. cat_variables contain the category column names.

cat_variables=list(lead_df.select_dtypes('object').columns)
for i in range(len(cat_variables)):
    if len((lead_df[cat_variables[i]].value_counts().index)) != 2:
        print(lead_df[cat_variables[i]].value_counts())
        print("\n")
        print("-------------------")


# In[28]:


#Dropping "'What matters most to you in choosing a course as there is only one valid value "better career prospects""

lead_df.drop(['What matters most to you in choosing a course'],axis=1,inplace=True)


# In[29]:


# Accessing country if it should be dropped
lead_df["Country"]  = lead_df.Country.apply(
    lambda x: "other countries" if (x != 'india' and x != "unknown") else x) # mapping the values
lead_df.Country.value_counts()/lead_df.shape[0] # calculating the percentage of each value present in the column


# In[30]:


#Dropping "'Country' as India seem like the only main value. Other countries correspond to only 0.03 %"

lead_df.drop(['Country'],axis=1,inplace=True)


# ## **Step 2. EDA**

# ### Step 2.1. Univariate Analysis

# #### 2.1.1. Categorical Variables

# In[31]:


#Plotting all category variables

plt.figure(figsize = (20,7))

eda_uni_cols=list(lead_df.select_dtypes('object').columns)
removelist=['Do Not Email', 'Do Not Call', 'Search', 'Newspaper Article', 'X Education Forums', 'Newspaper', 'Digital Advertisement', 'Through Recommendations', 'A free copy of Mastering The Interview','Lead Source','Last Activity','Specialization','Last Notable Activity']

# removing columns those are in the e as well as eda_uni_cols list from the eda_uni_cols list 
for e in removelist:
    if e in (eda_uni_cols):
        eda_uni_cols.remove(e)

for i in range(len(eda_uni_cols)):
    plt.subplot(1,2,i+1) # subplot of 1 x 2
    sns.countplot(lead_df[eda_uni_cols[i]]) # ploting the count plot of vaiables
    plt.title(eda_uni_cols[i]) # printing the title of graph


# ### **Observation:** 
# 
# - X Education recived, mostly the lead of customer from api and landing page submission
# - In the second graph, the custers of x Education are unemployeed and unworking

# In[32]:


plt.figure(figsize = (30,60))

eda_uni_cols=list(['Lead Source','Last Activity','Specialization','Last Notable Activity']) # the list of columns 

for i in range(len(eda_uni_cols)):
    plt.subplot(4,1,i+1) # subplot of 4 x 1
    sns.countplot(lead_df[eda_uni_cols[i]]).tick_params(axis='x', rotation = 45) # ploting the count plot of variables
    plt.title(eda_uni_cols[i]) # tittle of the plot


# ### **Observation:**
# 
# - In the first plot, it shows that the higest lead source are visiting the X Education from the "Google" and "direct Traffic".
# 
# - In the second plot, it shows that the higest lead are activated at the email or sms. 
# 
# - In the third plot, it shows that the higest lead specilization are unknown. That means lead havent provided thire interested specilization.
# 
# - In the fourth plot, it shows that the higest lead last notable activity are modified,email opend and sms sent. 

# In[33]:


c = list(lead_df.select_dtypes('object').columns)              # extracting only object data type variable and storing them into c class variable
new_list = []                                                  # defined new list
for i in range(len(c)):
    if len((lead_df[c[i]].value_counts().index)) == 2:         # checking the columns those index are equal to the two
        new_list.append(c[i])                                  # Storing the columns name into the new list

plt.figure(figsize=(30,20))                                    # defined the figure size 

for i in range(len(new_list)):
    plt.subplot(3,3,i+1)                                       # subplot of 5 x 2     
    sns.countplot(x=new_list[i],data=lead_df,hue='Converted')  # ploting the count plot of variables
    plt.title(new_list[i])                                     # define the title of plot
plt.show()


# ### **Observation:**
# 
# - In the first graph, it shows that most customer wants to recived the email from the x education Company. Also those custemer recived the email are more converted.
# 
# - In the first graph, it shows that most customer not wants to recived the A free copy of mastering the interview from the x education Company.
# 

# #### 2.1.2. Numerical Variables

# In[34]:


plt.figure(figsize = (20,10))
plt.subplot(2,2,1)
sns.distplot(lead_df['TotalVisits'], bins = 200)
plt.title('Total Visits')
plt.xlim(0,25)

plt.subplot(2,2,2)
sns.distplot(lead_df['Total Time Spent on Website'], bins = 10)
plt.title('Total Time Spent on Website')

plt.subplot(2,2,3)
sns.distplot(lead_df['Page Views Per Visit'], bins = 20)
plt.title('Page Views Per Visit')
plt.xlim(0,20)
plt.show()


# #### 2.1.3. Checking Lead variables against converted target variable

# In[35]:


plt.figure(figsize=(30,5))
plt.suptitle("Lead Variables Against Target")
plt.subplot(1,2,1)
sns.countplot(x='Lead Origin',hue='Converted',data=lead_df).tick_params(axis='x',rotation=90)
plt.title('Lead Origin')

plt.subplot(1,2,2)
sns.countplot(x='Lead Source',hue='Converted',data=lead_df).tick_params(axis='x',rotation = 90)
plt.title('Lead Source')
plt.show()


# In[36]:


plt.figure(figsize = (10,5))

sns.countplot(x='Last Activity', hue='Converted', data= lead_df).tick_params(axis='x', rotation = 90)
plt.title('Last Activity')
plt.show()


# In[37]:


plt.figure(figsize = (30,7))

plt.subplot(1,2,1)
sns.countplot(x='Specialization', hue='Converted', data= lead_df).tick_params(axis='x', rotation = 90)
plt.title('Specialization')

plt.subplot(1,2,2)
sns.countplot(x='What is your current occupation', hue='Converted', data= lead_df).tick_params(axis='x', rotation = 90)
plt.title('What is your current occupation')
plt.show()


# In[38]:


#Numeric Value correlation
plt.figure(figsize=[10,5])
plt.title("Numeric value Correlation")
sns.heatmap(lead_df.corr(),annot=True,cmap="Greens")
plt.show()


# ## Step 3: Outlier Treatment

# In[39]:


# cheking the outliers of only numeric values variables
numeric = lead_df[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]
# describing the variables in following percentiles 
numeric.describe(percentiles=[0.25,0.5,0.75,0.9,0.99])


# In[40]:


# Graphical view of the outliers

cols=['TotalVisits','Total Time Spent on Website','Page Views Per Visit'] # list of columns

plt.figure(figsize = (30,5))

plt.subplot(1,3,1)
sns.boxplot(lead_df['TotalVisits']) 
plt.subplot(1,3,2)
sns.boxplot(lead_df['Total Time Spent on Website'])
plt.subplot(1,3,3)
sns.boxplot(lead_df['Page Views Per Visit'] )

plt.show()


# ## <font color='green'> Retaining 99% quantile of data </font>

# In[41]:


outlier_cap_percent =.99 # define the cap of percentiles

# applying the treatment
for c in cols:
    lead_df=lead_df[lead_df[c] <= lead_df[c].quantile(outlier_cap_percent)]


# ## Step 4. Dummy Variables

# <font color='green'> Identifying all categorical variables

# In[42]:


# storing the variable, those data type is object
dummy_cols=lead_df.loc[:, lead_df.dtypes == 'object'].columns


# In[43]:


# Create dummy variables using the 'get_dummies function and storing into the dummy datafram
dummy = pd.get_dummies(lead_df[dummy_cols], drop_first=True)

# merege the results to the final dataframe
lead_df_dum = pd.concat([lead_df, dummy], axis=1)


# In[44]:


#dropping repeating columns
lead_df_dum.drop(dummy_cols, 1, inplace = True) # Dropping the dummy variables
# Dropping the not usable columns
lead_df_dum.drop(['What is your current occupation_unknown','Specialization_unknown'],axis=1,inplace=True)
lead_df_dum.head()


# In[45]:


# Cheking the dataframe info
lead_df_dum.info()


# ## 4. Test-Train Split

# In[46]:


# Import the required library
from sklearn.model_selection import train_test_split


# In[47]:


# dropping the target variable and unique id of the lead
X = lead_df_dum.drop(['Converted','Lead Number'], 1) # storing the dataframe into the X
X.head()


# In[48]:


# Putting the target variable in y
y = lead_df_dum['Converted']
y.head()


# In[49]:


# Split the dataset into 70% and 30% for train and test respectively
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=10)


# In[50]:


# Import MinMaxscaler
from sklearn.preprocessing import MinMaxScaler
# Scale the numeric features
scaler = MinMaxScaler()
X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])
X_train.head()


# In[51]:


# To check the correlation among varibles
plt.figure(figsize=(20,30))
sns.heatmap(X_train.corr())
plt.show()


# ### Observation: 
# - Its not easy to determine the correlation using above heatmap. Hence we are using RFE to determine the columns to be considered for the model.

# ## 5. Model Building 

# In[52]:


# Import 'LogisticRegression'
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[53]:


# Import RFE
from sklearn.feature_selection import RFE


# In[54]:


# Running RFE with 15 variables as output
rfe = RFE(logreg, 15)
# applying the fit on the x and y train dataframe
rfe = rfe.fit(X_train, y_train)


# In[55]:


# Features that have been selected by RFE
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[56]:


# Put all the columns selected by RFE in the variable 'col'
col = X_train.columns[rfe.support_]


# In[57]:


col


# In[58]:


# Selecting columns selected by RFE
X_train = X_train[col]


# In[59]:


# Importing statsmodels
import statsmodels.api as sm


# In[60]:


# Importing 'variance_inflation_factor'
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[61]:


# Make a VIF dataframe for all the variables present.
# Creating the function of the VIF to called multiple time
def vif_():
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif


# ## **5.1 Model One**

# In[62]:


X_train_sm = sm.add_constant(X_train)                                   # adding the constant
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())    # applying the GLM modeling
res = logm1.fit()                                                       # apply the fit on the model 
print(res.summary())                                                    # printing the summary of model
print(vif_())                                                           # printing the VIF of model


# <font color='Red'>**The vif of "What is your current occupation_housewife" variable is good but the "P value" is greater than the 0.05. Due to this we are dropping this variable from the model**

# In[63]:


# dropping the variable
X_train.drop('What is your current occupation_housewife', axis = 1, inplace = True)


# ## **5.2 Model Two**

# In[64]:


X_train_sm = sm.add_constant(X_train)                                   # adding the constant
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())    # applying the GLM modeling
res = logm2.fit()                                                       # apply the fit on the model 
print(res.summary())                                                    # printing the summary of model
print(vif_())                                                           # printing the VIF of model


# <font color='Red'>**The vif of "TotalVisits" variable is not good but the "P value" is less than the 0.05. Due to this we are dropping this variable from the model**

# In[65]:


# Dropping the variable
X_train.drop('TotalVisits', axis = 1, inplace = True)


# ## **5.3 Model Three**

# In[66]:


X_train_sm = sm.add_constant(X_train)                                   # adding the constant
logm3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())    # applying the GLM modeling
res = logm3.fit()                                                       # apply the fit on the model 
print(res.summary())                                                    # printing the summary of model
print(vif_())                                                           # printing the VIF of model


# ## Observation:
# 
# ### <font color= green>**All the VIF values are good and all the p-values are below 0.05. It is, therfore, the model three is our final model.**</font>

# ## Step 6. Creating Prediction

# In[67]:


# Predicting the probabilities on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[68]:


# Reshaping to an array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[69]:


# Data frame with given convertion rate and probablity of predicted ones
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Convert_Prob':y_train_pred})
y_train_pred_final.head()


# In[70]:


# Substituting 0 or 1 with the cut off as 0.8
y_train_pred_final['Predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.8 else 0)
y_train_pred_final.head()


# ##  Step 7. Model Evaluation

# In[71]:


# Importing metrics from sklearn for evaluation
from sklearn import metrics


# In[72]:


# Creating confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[73]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# <font color= green>___That's around 74% accuracy with is a  good value___</font>

# In[74]:


# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]


# In[75]:


# Calculating the sensitivity
TP/(TP+FN)


# In[76]:


# Calculating the specificity
TN/(TN+FP)


# <font color= green>___With the current cut off as 0.5 we have around 81% accuracy, sensitivity of around 70% and specificity of around 87%.___</font>

# ## Step 8. Optimise Cut off (ROC Curve)

# The previous cut off was randomely selected. Now to find the optimum one

# In[77]:


# ROC function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[78]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Convert_Prob, drop_intermediate = False )


# In[79]:


# Call the ROC function
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Convert_Prob)


# <font color= green>___The area under ROC curve is 0.87 which is a very good value.___</font>

# In[80]:


# Creating columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()


# In[81]:


# Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
# Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df


# In[82]:


# Plotting it
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# <font color= green>___From the graph it is visible that the optimal cut off is at 0.35.___</font>

# In[83]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Convert_Prob.map( lambda x: 1 if x > 0.35 else 0)
y_train_pred_final.head()


# In[84]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[85]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[86]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[87]:


# Calculating the sensitivity
TP/(TP+FN)


# In[88]:


# Calculating the specificity
TN/(TN+FP)


# ## <font color= green>___With the current cut off as 0.40 we have accuracy and specificity of around 80%. Also, we have the sensitivity is 75%___</font>

# ## 9. Precision-Recall

# In[89]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
confusion


# In[90]:


# Precision = TP / TP + FP
confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[91]:


#Recall = TP / TP + FN
confusion[1,1]/(confusion[1,0]+confusion[1,1])


# ### 9.1. Precision and recall tradeoff

# In[92]:


from sklearn.metrics import precision_recall_curve


# In[93]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[94]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Convert_Prob)


# In[95]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[96]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.40 else 0)
y_train_pred_final.head()


# In[97]:


# Accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[98]:


# Creating confusion matrix again
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[99]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[100]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[101]:


#Recall = TP / TP + FN
TP / (TP + FN)


# In[ ]:





# ## Step 10: Making predictions on the test set

# In[102]:


X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[103]:


col = X_train.columns
X_test = X_test[col]
X_test.head()


# In[104]:


X_test_sm = sm.add_constant(X_test)


# Making predictions on the test set

# In[105]:


y_test_pred = res.predict(X_test_sm)


# In[106]:


y_test_pred[:10]


# In[107]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[108]:


# Let's see the head
y_pred_1.head()


# In[109]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[110]:


# Putting CustID to index
y_test_df['Index_ID'] = y_test_df.index


# In[111]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[112]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[113]:


y_pred_final.head()


# In[114]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Convert_Prob'})


# In[115]:


# Rearranging the columns
y_pred_final = y_pred_final.reindex(['Index_ID','Converted','Convert_Prob'], axis=1)


# In[116]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[117]:


y_pred_final['final_predicted'] = y_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.35 else 0)


# In[118]:


y_pred_final.shape


# In[119]:


y_pred_final.head()


# In[120]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)


# In[121]:


#Creating confusion Matrix
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion2


# In[122]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[123]:


# Let us calculate specificity
TN / float(TN+FP)


# In[124]:


lead_df_copy=lead_df.copy()
lead_df_copy['Index_ID']=lead_df_copy.index


# In[125]:


final_lead = pd.merge(lead_df_copy[['Lead Number','Index_ID']], y_pred_final, on = 'Index_ID')
final_lead.head()


# In[126]:


final_lead['lead score']=round(final_lead.Convert_Prob*100,0)


# In[127]:


final_lead[final_lead.final_predicted==1]


# In[128]:


len(final_lead[final_lead.Converted == 1])/len(final_lead[final_lead.final_predicted == 1])


# # **Summary**
# 
# - **The conclusion of this report, the X Education company has to focus mostly on the following variable to achieve the 80% lead conversion rate toward their company. These variables are very potential to understand the customer profile and weather that customer potentially will buy the courses from the company or not. It is, therefore, the variables are:**
# 
# 
# - **1. Total time Spent on website.**
# - **2. Whenever the Lead Origin was:**
#                                  2.1. Add format 
# - **3. Whenever the Lead Source was:**
#                                  3.1. Direct Traffic
#                                  3.2. Google
#                                  3.3. Organic Search
#                                  3.4. Referral Sites
#                                  3.5. Welingak Website
# - **4. When last notable activity:**
#                                  4.1. Olark Chat Conversation
#                                  4.2. Unreachable
# - **5. When the email does not send yes.** 
# - **6. When the Last Activity:**
#                                  6.1 Had a Phone Conversation
#                                  6.2 SMS Sent
# 
# - **7. When the customer current occupation is working professional.**
# 
