#!/usr/bin/env python
# coding: utf-8

# # EDA Case Study 

# ## Team : Shivram Jayakumar & Pappu Dindayal Kapgate

# In[1]:


# Importing all the neccessary libraries
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Read Application Data
appl_data = pd.read_csv("application_data.csv", header=0)

# Previous application data
p_appl_data = pd.read_csv("previous_application.csv",header=0)


# # <font color='Green'>**1. Exploring the Data in Application Data file</font>**

# In[3]:


appl_data.head()


# In[4]:


#Checking number of rows and columns in the file

appl_data.shape


# In[5]:


#Check the column data size and data types
appl_data.info()

#This is a lot of data!!!


# In[6]:


#checking statistical data 

appl_data.describe()


# # <font color='Green'>**2. Identfying the missing values**

# #### **2.1 Identify the % of missing values and list greater than 25% null value columns**

# In[7]:


# Identify the % of missing values and list greater than 25% null value columns

null_cols=((appl_data.isnull().sum()*100)/appl_data.shape[0]).round(2)
null_cols


# ### **2.2 Drop columns which have more than 25% of Null values in them**

# In[8]:


appl_data.loc[:,appl_data.isnull().mean()>=.25]


# In[9]:


#Drop the columns
appl_data.drop(appl_data.loc[:,appl_data.isnull().mean()>=.25],axis=1,inplace=True)


# In[10]:


#Once dropped the total number of columns are 72
appl_data.shape


# In[11]:


#Checking the dataset again
appl_data.head()


# ### **2.3 Checking Percentage of null values on the remaining columns which are less than 25% Null**

# In[12]:


null_cols=((appl_data.isnull().sum()*100)/len(appl_data))
null_cols[null_cols>0.0]


# # <font color='Green'> **3. Impute the Null values with mean, median, mode or 0  where applicable**

# ##### A) AMT_GOODS_PRICE

# In[13]:


# col => AMT_GOODS_PRICE , impute by mean as it is a numeric value of goods price. So in a larger population mean would be the best fit.

appl_data.AMT_GOODS_PRICE.fillna(appl_data.AMT_GOODS_PRICE.mean(), inplace = True)


# In[14]:


#No null values => AMT_GOODS_PRICE
appl_data.AMT_GOODS_PRICE.isnull().value_counts()


# ##### B) NAME_TYPE_SUITE

# In[15]:


# col => NAME_TYPE_SUITE, looks like Unaccompanied is most occurring value for this field.
appl_data.NAME_TYPE_SUITE.value_counts()


# In[16]:


#Because it a categorical variable, we fill null values with mode - Unaccompanied as it occurs the most in the list.
appl_data['NAME_TYPE_SUITE'].fillna(appl_data.NAME_TYPE_SUITE.mode()[0], inplace=True)


# In[17]:


appl_data.NAME_TYPE_SUITE.isnull().value_counts()


# ##### C) EXT_SOURCE_2  and EXT_SOURCE_3

# In[18]:


#Not sure what these columns are used for. Will determine later.

cols_to_change=['EXT_SOURCE_2','EXT_SOURCE_3']
appl_data[cols_to_change].describe()


# In[19]:


#Filling NA with mean value
appl_data[cols_to_change]=appl_data[cols_to_change].fillna(appl_data[cols_to_change].mean()) 


# In[20]:


appl_data.EXT_SOURCE_2.isnull().value_counts()


# In[21]:


appl_data.EXT_SOURCE_3.isnull().value_counts()


# ##### D) CNT_FAM_MEMBERS

# In[22]:


appl_data.CNT_FAM_MEMBERS.isnull().sum()


# In[23]:


appl_data.CNT_FAM_MEMBERS= appl_data.CNT_FAM_MEMBERS.fillna(0) 


# ##### E) Clean OBS_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE, DEF_30_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE 

# In[24]:


cols_to_change= ['OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE']


# In[25]:


appl_data[cols_to_change].describe()


# In[26]:


appl_data.DEF_60_CNT_SOCIAL_CIRCLE.value_counts()


# In[27]:


appl_data.DEF_30_CNT_SOCIAL_CIRCLE.value_counts()


# In[28]:


appl_data.OBS_60_CNT_SOCIAL_CIRCLE.value_counts()


# In[29]:


appl_data.OBS_30_CNT_SOCIAL_CIRCLE.value_counts()


# In[30]:


# looks like these are integers stored as floats. So along with fixing the nulls, lets also change the datatypes.
appl_data[cols_to_change].dtypes


#    ##### E.1 Change datatype from float to int

# In[31]:


appl_data[cols_to_change]=appl_data[cols_to_change].astype('Int64',errors='ignore')


# In[32]:


#Changed data types
appl_data[cols_to_change].dtypes


# In[33]:


plt.rcParams.update({'font.size': 10})
plt.figure(figsize=[10,10])
appl_data[cols_to_change].plot.box(vert=False)
plt.show()


# In[34]:


appl_data[cols_to_change].describe()


# In[35]:


#Would make sense to apply mean on the social circle counts as outliers dont affect the mean that much.

appl_data[cols_to_change]=appl_data[cols_to_change].fillna(round(appl_data[cols_to_change].mean()))


# In[36]:


#Check if all null values have been replaced.

appl_data[cols_to_change].isnull().sum()


# #### F) AMT_REQ_CREDIT_BUREAU

# In[37]:


#Number of enquiries to Credit Bureau

cols_to_change= ['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
           'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']


# In[38]:


appl_data[cols_to_change].dtypes


# In[39]:


appl_data.AMT_REQ_CREDIT_BUREAU_HOUR.value_counts()


# In[40]:


appl_data.AMT_REQ_CREDIT_BUREAU_DAY.value_counts()


# In[41]:


appl_data.AMT_REQ_CREDIT_BUREAU_WEEK.value_counts()


# In[42]:


appl_data.AMT_REQ_CREDIT_BUREAU_MON.value_counts()


# In[43]:


#convert all the AMT_REQ columns to integer as they contain whole numbers

appl_data[cols_to_change]=appl_data[cols_to_change].astype('Int64',errors='ignore')


# In[44]:


appl_data[cols_to_change].dtypes


# In[45]:


appl_data[cols_to_change].describe()

#Apart from AMT_REQ_CREDIT_BUREAU_YEAR all the other columns have mean values and 50-75 percentile closer to 0.


# In[46]:


#Fill na for all the amount columns to 0 except AMT_REQ_CREDIT_BUREAU_YEAR

appl_data.AMT_REQ_CREDIT_BUREAU_HOUR.fillna(value = 0, inplace = True)
appl_data.AMT_REQ_CREDIT_BUREAU_DAY.fillna(value = 0, inplace = True)
appl_data.AMT_REQ_CREDIT_BUREAU_WEEK.fillna(value = 0, inplace = True)
appl_data.AMT_REQ_CREDIT_BUREAU_MON.fillna(value = 0, inplace = True)
appl_data.AMT_REQ_CREDIT_BUREAU_QRT.fillna(value = 0, inplace = True)


# In[47]:


plt.rcParams.update({'font.size': 14})
appl_data.AMT_REQ_CREDIT_BUREAU_YEAR.plot.box()


# In[48]:


#Fill Null values for AMT_REQ_CREDIT_BUREAU_YEAR column with median

appl_data.AMT_REQ_CREDIT_BUREAU_YEAR.fillna(value = appl_data.AMT_REQ_CREDIT_BUREAU_YEAR.median(), inplace = True)


# #### G. AMT_ANNUITY
# 
# 

# In[49]:


appl_data.AMT_ANNUITY.describe()


# In[50]:


appl_data.AMT_ANNUITY.fillna(value = appl_data.AMT_ANNUITY.mean(), inplace = True)


# #### H. DAYS_LAST_PHONE_CHANGE

# In[51]:


appl_data.DAYS_LAST_PHONE_CHANGE.describe()


# In[52]:


appl_data.DAYS_LAST_PHONE_CHANGE.fillna(value = 0, inplace = True)


# In[53]:


#Final check for null value columns. All columns have been treated.

null_cols=((appl_data.isnull().sum()*100)/len(appl_data))
null_cols[null_cols>0.0]


# # <font color='Green'> **4 Check Data Types And Convert To Appropriate Datatype**

# In[54]:


appl_data.info()


# In[55]:


#Check if any Objects need to be converted. Looks good!
appl_data.select_dtypes('object')


# In[56]:


#Check if any float columns need to be converted. Looks like some can be converted into integers!
appl_data.select_dtypes('float')


# ### 4.1 Datatype change

# In[57]:


#Counts and days are whole numbers in real life, so converting them into integers
# 'DAYS_LAST_PHONE_CHANGE','CNT_FAM_MEMBERS','DAYS_REGISTRATION'

cols_to_change=['CNT_FAM_MEMBERS','DAYS_LAST_PHONE_CHANGE','DAYS_REGISTRATION']
appl_data[cols_to_change].dtypes


# In[58]:


appl_data.DAYS_REGISTRATION.value_counts()


# In[59]:


#Found a floating point value for days_registration. 

appl_data[appl_data.DAYS_REGISTRATION.apply(lambda x: False if x.is_integer() else True)].DAYS_REGISTRATION


# In[60]:


#drop the record!!!

appl_data=appl_data[~appl_data.DAYS_REGISTRATION.apply(lambda x: False if x.is_integer() else True)]


# In[61]:


appl_data.CNT_FAM_MEMBERS.value_counts()


# In[62]:


appl_data.DAYS_LAST_PHONE_CHANGE.value_counts()


# In[63]:


#convert the three columns into integers

appl_data[cols_to_change]=appl_data[cols_to_change].astype('Int64',errors='ignore')


# In[64]:


appl_data[cols_to_change].dtypes


# ### 4.2 Dealing with flags (Y/N) - Binary variables

# In[65]:


appl_data.filter(regex='FLAG')


# In[66]:


flag_columns=list(appl_data.filter(regex='FLAG'))
flag_columns.remove('FLAG_OWN_CAR') # This is already in Y/N format
flag_columns.remove('FLAG_OWN_REALTY') # This is already in Y/N format
flag_columns


# In[67]:


#converting all flag columns to Y or N instead of 1 and 0

appl_data[flag_columns]=appl_data[flag_columns].replace((0, 1), ('N', 'Y'))


# In[68]:


#All the flag have been converted to Y and N

appl_data[list(appl_data.filter(regex='FLAG'))]


# In[ ]:





# # <font color='Green'> **5 Identifying and dealing with Outliers**

# ### **5.1 Cleaning Float columns**

# In[69]:


#Build a box plot for all the relevant numeric columns

appl_data.select_dtypes('float64')

#Lets pick AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, REGION_POPULATION_RELATIVE


# In[70]:


#No clue what this EXT_SOURCE_2 and EXT_SOURCE_3 , hence removing it. Couldn't understand from the description sheet.

appl_data = appl_data.drop(['EXT_SOURCE_2','EXT_SOURCE_3'],axis=1)
    


# In[71]:


float_cols=appl_data.select_dtypes('float64').columns


# In[72]:


#Lets look at the float outliers
#AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, REGION_POPULATION_RELATIVE
plt.suptitle("Float Outliers")
plt.rcParams.update({'font.size': 14})
for col in float_cols:
    f, a = plt.subplots(1,1,figsize=(12,3))
    plt.title(col,color="blue")
    sns.boxplot(appl_data[col], color="green")
    plt.show()


# ### **5.1.1 Fixing outliers for float columns**

# #### **A) AMT_INCOME_TOTAL**

# In[73]:


#Keeping data below 95 percentile

appl_data=appl_data[appl_data.AMT_INCOME_TOTAL < appl_data.AMT_INCOME_TOTAL.quantile(0.95)]


# In[74]:


plt.rcParams.update({'font.size': 10})
sns.boxplot(appl_data.AMT_INCOME_TOTAL,color="green")


# #### **B) AMT_CREDIT**

# In[75]:


# Keeping only 99 percentile of data
plt.rcParams.update({'font.size': 14})
appl_data=appl_data[appl_data.AMT_CREDIT < appl_data.AMT_CREDIT.quantile(0.99)]


# In[76]:


plt.figure(figsize=[10,2])
sns.boxplot(appl_data.AMT_CREDIT,color="green")
plt.show()


# #### **C) Fixing AMT_ANNUITY**

# In[77]:


# Keeping only 99 percentile of data

appl_data=appl_data[appl_data.AMT_ANNUITY < appl_data.AMT_ANNUITY.quantile(0.99)]


# In[78]:


plt.figure(figsize=[10,2])
plt.rcParams.update({'font.size': 14})
sns.boxplot(appl_data.AMT_ANNUITY,color="green")
plt.show()


# #### **D) Fixing AMT_GOOD_PRICE**

# In[79]:


# Keeping only 90 percentile of data
appl_data=appl_data[appl_data.AMT_GOODS_PRICE < appl_data.AMT_GOODS_PRICE.quantile(0.90)]


# In[80]:


plt.figure(figsize=[10,2])
plt.rcParams.update({'font.size': 14})
sns.boxplot(appl_data.AMT_GOODS_PRICE,color="green")
plt.show()


# In[ ]:





# ### **5.2 Cleaning Integer columns**

# In[81]:


#List of all integer columns
appl_data.select_dtypes('Int64')


# ### **5.2.1 Fixing outliers for integer columns**

# In[82]:


#Lets fist pick CNT_CHILDREN, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_ID_PUBLISH, DAYS_REGISTRATION, CNT_FAM_MEMBERS, DAYS_LAST_PHONE_CHANGE

plt.figure(figsize=[25,10])
plt.rcParams.update({'font.size': 14})
plt.subplot(2,4,1)
sns.boxplot(appl_data.CNT_CHILDREN,color="green")
plt.subplot(2,4,2)
sns.boxplot(appl_data.CNT_FAM_MEMBERS,color="green")
plt.subplot(2,4,3)
sns.boxplot(appl_data.DAYS_BIRTH,color="green")
plt.subplot(2,4,4)
sns.boxplot(appl_data.DAYS_ID_PUBLISH,color="green")
plt.subplot(2,4,5)
sns.boxplot(appl_data.DAYS_REGISTRATION,color="green")
plt.subplot(2,4,6)
sns.boxplot(appl_data.DAYS_EMPLOYED,color="green")
plt.subplot(2,4,7)
sns.boxplot(appl_data.DAYS_LAST_PHONE_CHANGE,color="green")
plt.show()


# #### **A) CNT_Children**

# In[83]:


#CNT_Children, CNT_FAM_MEMBERS needs to be cleaned for outliers and covert DAY columns into years or months.

appl_data.CNT_CHILDREN.describe()

# Max number of children seems wrong


# In[84]:



plt.figure(figsize=[25,10])
plt.rcParams.update({'font.size': 14})
plt.subplot(2,2,1)
plt.title("CNT_CHILDREN")
appl_data.CNT_CHILDREN.plot.hist(color="green")
plt.subplot(2,2,2)
sns.boxplot(appl_data.CNT_CHILDREN,color="red")
plt.show()


# In[85]:


appl_data.CNT_CHILDREN.max() #This seems like a outlier that we can remove


# In[86]:


appl_data.CNT_CHILDREN.quantile(.99) #3 which is more close to reality


# In[87]:


appl_data=appl_data[appl_data.CNT_CHILDREN<=appl_data.CNT_CHILDREN.quantile(.99)] #Droping all the rows with greater than 3 children


# In[88]:


appl_data.shape


# #### **B) CNT_FAM_MEMBERS**

# In[89]:


#family Members count

plt.figure(figsize=[25,10])
plt.rcParams.update({'font.size': 14})
plt.subplot(2,2,1)
plt.title("CNT_FAM_MEMBERS")
appl_data.CNT_FAM_MEMBERS.plot.hist(color="green")
plt.subplot(2,2,2)
sns.boxplot(appl_data.CNT_FAM_MEMBERS,color="red")
plt.show()

#Looks like when we removed the children outlier records, even most of cnt_fam_member outliers got fixed.


# In[ ]:





# #### **C) DAYS_EMPLOYED**

# In[90]:


#Fixing DAYS_EMPLOYED


plt.figure(figsize=[25,10])
plt.rcParams.update({'font.size': 14})
plt.title("DAYS_EMPLOYED")
plt.subplot(2,2,1)
plt.title("DAYS_EMPLOYED")
appl_data.DAYS_EMPLOYED.plot.hist(color="green")
plt.subplot(2,2,2)
plt.title("DAYS_EMPLOYED")
sns.boxplot(appl_data.DAYS_EMPLOYED,color="red")
plt.show()


# In[91]:


appl_data.DAYS_EMPLOYED.describe()
#looks like majority of the data is negative, does that mean they are unemployed or is it a data error ?


# In[92]:


#Need to clean up data 
appl_data.DAYS_EMPLOYED.quantile(.81)


# In[93]:


# lets get the absoulute value of days employed
appl_data.DAYS_EMPLOYED=appl_data.DAYS_EMPLOYED.apply(lambda x: abs(x))


# In[94]:


#What if we take only 75% of data for this columns ?
appl_data[appl_data.DAYS_EMPLOYED<appl_data.DAYS_EMPLOYED.quantile(.75)].DAYS_EMPLOYED.plot.box(color="green")
plt.rcParams.update({'font.size': 14})
plt.show()


# In[95]:


appl_data=appl_data[appl_data.DAYS_EMPLOYED<appl_data.DAYS_EMPLOYED.quantile(.75)]


# In[96]:


appl_data.DAYS_EMPLOYED.plot.box(color="green")
plt.rcParams.update({'font.size': 14})
plt.show()


# #### **Leveraging days employeed to create year_employed.**

# In[97]:


appl_data["YEARS_EMPLOYED"]= appl_data.DAYS_EMPLOYED.apply(lambda x: x/365)


# In[98]:


appl_data.YEARS_EMPLOYED.plot.hist(color="green")
plt.rcParams.update({'font.size': 14})


# ### **D) Fixing other DAYS columns similarly**

# In[99]:


# Converting all days fields to absolute values

appl_data.DAYS_BIRTH=appl_data.DAYS_BIRTH.apply(lambda x: abs(x))
appl_data.DAYS_ID_PUBLISH=appl_data.DAYS_ID_PUBLISH.apply(lambda x: abs(x))
appl_data.DAYS_REGISTRATION=appl_data.DAYS_REGISTRATION.apply(lambda x: abs(x))
appl_data.DAYS_LAST_PHONE_CHANGE=appl_data.DAYS_LAST_PHONE_CHANGE.apply(lambda x: abs(x))


# In[100]:


appl_data=appl_data[appl_data.DAYS_REGISTRATION<appl_data.DAYS_REGISTRATION.quantile(.90)]
appl_data=appl_data[appl_data.DAYS_LAST_PHONE_CHANGE<appl_data.DAYS_LAST_PHONE_CHANGE.quantile(.90)]


# In[101]:


plt.figure(figsize=[25,10])
plt.rcParams.update({'font.size': 14})
plt.subplot(2,3,1)
sns.boxplot(appl_data.DAYS_BIRTH,color="green")
plt.subplot(2,3,2)
sns.boxplot(appl_data.DAYS_ID_PUBLISH,color="green")
plt.subplot(2,3,3)
sns.boxplot(appl_data.DAYS_REGISTRATION,color="green")
plt.subplot(2,3,4)
sns.boxplot(appl_data.DAYS_EMPLOYED,color="green")
plt.subplot(2,3,5)
sns.boxplot(appl_data.DAYS_LAST_PHONE_CHANGE,color="green")
plt.show()


# In[102]:


# Converting days into years

appl_data["YEARS_BIRTH"]= appl_data.DAYS_BIRTH.apply(lambda x: x/365)
appl_data["YEARS_ID_PUBLISH"]= appl_data.DAYS_ID_PUBLISH.apply(lambda x: x/365)
appl_data["YEARS_REGISTRATION"]= appl_data.DAYS_REGISTRATION.apply(lambda x: x/365)
appl_data["YEARS_LAST_PHONE_CHANGE"]= appl_data.DAYS_LAST_PHONE_CHANGE.apply(lambda x: x/365)


# In[103]:




plt.figure(figsize=[25,10])
plt.rcParams.update({'font.size': 14})
plt.subplot(2,2,1)
plt.title("YEARS_BIRTH")
appl_data["YEARS_BIRTH"].plot.hist(color="green")
plt.subplot(2,2,2)
plt.title("YEARS_ID_PUBLISH")
appl_data["YEARS_ID_PUBLISH"].plot.hist(color="green")
plt.subplot(2,2,3)
plt.title("YEARS_REGISTRATION")
appl_data["YEARS_REGISTRATION"].plot.hist(color="green")
plt.subplot(2,2,4)
plt.title("YEARS_LAST_PHONE_CHANGE")
appl_data["YEARS_LAST_PHONE_CHANGE"].plot.hist(color="green")
plt.show()


# In[ ]:





# # <font color='Green'> **6. Create Categorical Variable**

# ### **A) AMT_INCOME**

# In[104]:


#creating categories for customer incomes.

label = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
appl_data["AMT_INCOME_CATEG"]= pd.qcut(appl_data.AMT_INCOME_TOTAL,q=[0, .2, .4, .6, .8, 1],labels=label)
appl_data.AMT_INCOME_CATEG.value_counts()


# ### **B) AGE_CUSTOMER**

# In[105]:


# Converting customer Birth days into age
appl_data['AGE_CUSTOMER'] = round((appl_data[['DAYS_BIRTH']] /365))


# In[106]:


# data type- float to int
appl_data['AGE_CUSTOMER'] = appl_data['AGE_CUSTOMER'].astype('Int64',errors='ignore')


# In[107]:


appl_data[appl_data.AGE_CUSTOMER<0][['AGE_CUSTOMER','DAYS_BIRTH']]


# In[108]:


sns.boxplot(appl_data.AGE_CUSTOMER,color='green')
plt.rcParams.update({'font.size': 14})
plt.show()


# In[109]:


# Creating age groups of customer
bins = [18, 30, 40, 50, 60, 70, 120]
age_group_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
appl_data['AGE_GROUP_CUSTOMER'] = pd.cut(appl_data.AGE_CUSTOMER,bins,labels=age_group_labels,include_lowest=True)
appl_data['AGE_GROUP_CUSTOMER'].value_counts()


# In[110]:


#validating the age data falls under the correct category

appl_data[['AGE_CUSTOMER','AGE_GROUP_CUSTOMER']].head()


# In[ ]:





# # <font color='Green'>**7. Performing Analysis**

# ## **7.1 Univariate Analysis**

# ### **7.1.1 Categorical unordered univariate analysis**

# #### **A) TARGET variable**

# In[111]:


# Creating new col with replacing values of target variable and spliting the data for ease!
appl_data['paystatus'] = appl_data.TARGET.replace([0,1],['On-Time','Late-Payments'])
on_time_appl_data = appl_data[appl_data.paystatus=='On-Time']
Late_Payments_appl_data = appl_data[appl_data.paystatus=='Late-Payments']


# In[112]:


sns.countplot(appl_data.paystatus)
plt.rcParams.update({'font.size': 14})
plt.xlabel("TARGET Value")
plt.ylabel("Count of Applications")
plt.title("Distribution of TARGET Variable")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>In the given population, the number of people having difficulties making payments is less than other applicants. The dataset is leaning towards ontime payments which may be due to random sampling.</font>

# #### **B) GENDER variable**

# In[113]:


# Count of Gender
plt.figure(figsize=[25,12])
plt.rcParams.update({'font.size': 14})
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(on_time_appl_data.CODE_GENDER)
plt.xlabel("Gender Type")
plt.ylabel("Count of Gender")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.CODE_GENDER)
plt.xlabel("Gender Type")
plt.ylabel("Count of Gender")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'> Females make more on-time payments than Male counterparts. But for late payments both genders are considerably similar. Hence female customers may be marginally better than male customers</font>

# #### **C) INCOME TYPE variable**

# In[114]:


on_time_appl_data.NAME_INCOME_TYPE.value_counts()


# In[115]:


# Count of NAME_INCOME_TYPE

plt.figure(figsize=[25,12])
plt.rcParams.update({'font.size': 10})
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(on_time_appl_data.NAME_INCOME_TYPE)
plt.xlabel("Income Type")
plt.ylabel("Count of Income")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.NAME_INCOME_TYPE)
plt.xlabel("Income Type")
plt.ylabel("Count of Income")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'> Student,Pensioner,Maternity Leave and Businessman are not present in the late payments data group. Also On-time paying customers are higher than late paying customers accross all income type groups.</font>

# #### **D) ORGANIZATION_TYPE Variable**

# In[116]:


# Count of ORGANIZATION_TYPE

plt.figure(figsize=[25,12])
plt.rcParams.update({'font.size': 10})
plt.subplot(2,2,1)
plt.title("On-Time Payments")

sns.countplot(on_time_appl_data.ORGANIZATION_TYPE)
plt.xticks( rotation='vertical')
plt.xlabel("Cumpany Type")
plt.ylabel("Count of Company Type")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.ORGANIZATION_TYPE)
plt.xticks( rotation='vertical')
plt.xlabel("Company Type")
plt.ylabel("Count of Company type")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'> No key observation to report, only counts are different</font>

# #### **E) FAMILY TYPE VAriable**

# In[117]:


plt.figure(figsize=[25,12])
plt.rcParams.update({'font.size': 14})
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(on_time_appl_data.NAME_FAMILY_STATUS)
plt.xticks( rotation='vertical')
plt.xlabel("Family Type")
plt.ylabel("Count of Family Type")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.NAME_FAMILY_STATUS)
plt.xticks( rotation='vertical')
plt.xlabel("Family Type")
plt.ylabel("Count of Family Type")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# ### **7.1.2 Categorical ordered univariate analysis**

# #### **A) YEARS_EMPLOYEED**

# In[118]:


appl_data.YEARS_EMPLOYED = appl_data.YEARS_EMPLOYED.apply(lambda x: round(x,0))


# In[119]:


appl_data.YEARS_EMPLOYED.value_counts()


# In[120]:


# Creating age groups of customer
bins = [0, 4, 8, 12, 16]
age_group_labels = ['0-4', '5-8', '9-12', '13-16']
appl_data['Years_Employeed_Range'] = pd.cut(appl_data.YEARS_EMPLOYED,bins,labels=age_group_labels,include_lowest=True)
appl_data['Years_Employeed_Range'].value_counts()
           


# In[121]:


# Reassigning the dataframe due to the new columns

on_time_appl_data = appl_data[appl_data.paystatus=='On-Time']
Late_Payments_appl_data = appl_data[appl_data.paystatus=='Late-Payments']

# Count of Years_Employeed_Range

plt.figure(figsize=[25,12])
plt.rcParams.update({'font.size': 14})
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(on_time_appl_data.Years_Employeed_Range)

plt.xticks( rotation='horizontal')
plt.xlabel("Employee Years Range Type")
plt.ylabel("Count of Years Range Type")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.Years_Employeed_Range)
plt.xticks( rotation='horizontal')
plt.xlabel("Employee Years Range Type")
plt.ylabel("Count of Years Range Type")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# #### **B) INCOME CATEGORY**

# In[122]:


plt.figure(figsize=[25,12])
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(on_time_appl_data.AMT_INCOME_CATEG)
plt.xticks( rotation='horizontal')
plt.xlabel("INCOME CATEGORY")
plt.ylabel("Number of customers")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.AMT_INCOME_CATEG)
plt.xticks( rotation='horizontal')
plt.xlabel("INCOME CATEGORY")
plt.ylabel("Number of customers")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# #### **C) Education Type**

# In[123]:


# Count of education of customer
plt.figure(figsize=[25,12])
plt.rcParams.update({'font.size': 9})
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(on_time_appl_data.NAME_EDUCATION_TYPE)
plt.xlabel("Education Type")
plt.ylabel("Count of educated customers")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.NAME_EDUCATION_TYPE)
plt.xlabel("Education Type")
plt.ylabel("Count of educated customers")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# #### **D) AGE GROUP CUSTOMER**

# In[124]:


plt.figure(figsize=[25,12])
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(on_time_appl_data.AGE_GROUP_CUSTOMER)
plt.xticks( rotation='horizontal')
plt.xlabel("AGE GROUP CUSTOMER")
plt.ylabel("Number of customers")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.AGE_GROUP_CUSTOMER)
plt.xticks( rotation='horizontal')
plt.xlabel("AGE GROUP CUSTOMER")
plt.ylabel("Number of customers")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# In[ ]:





# In[ ]:





# ## **7.2 Bivariate Analysis**

# ### **7.2.1 Two Numeric Variable Analysis**

# In[125]:


plt.figure(figsize=[25,12])
plt.subplot(2,2,1)
sns.scatterplot("AMT_CREDIT","AMT_GOODS_PRICE",data=appl_data,hue="paystatus",palette = ['red','green'])
plt.subplot(2,2,2)
sns.scatterplot("AMT_CREDIT","AMT_INCOME_TOTAL",data=appl_data,hue="paystatus",palette = ['red','green'])
plt.subplot(2,2,3)
sns.scatterplot("AMT_ANNUITY","AMT_GOODS_PRICE",data=appl_data,hue="paystatus",palette = ['red','green'])
plt.subplot(2,2,4)
sns.scatterplot("AMT_ANNUITY","AMT_CREDIT",data=appl_data,hue="paystatus",palette = ['red','green'])

plt.show()


# # <font color='red'>Observation:</font>
# ## <font color='white'>1. We can see a linear progression between credit amount,annuity amount and goods price. Even customers with payment diffculties have linearly progressed accross all the above plot. </font>
# ## <font color='white'>2. We didnt find strong linear corelation between income and credit amount. We were expecting to see people with higher income get more credit.</font>
# 
# 

# ## **B) YEARS_EMPLOYED**

# In[126]:


fig,ax=plt.subplots(figsize=[10,8])

plt.rcParams.update({'font.size': 14})
sns.kdeplot(appl_data[appl_data.TARGET==0].YEARS_EMPLOYED, shade=True, color="green", label="on-time", ax=ax)
sns.kdeplot(appl_data[appl_data.TARGET==1].YEARS_EMPLOYED, shade=True, color="red", label="Late payments", ax=ax)
ax.set_ylabel("Density",fontsize=18)
ax.set_xlabel("Years of Employment",fontsize=18)
fig.suptitle("Payment Status by Years of Employment")
plt.show()


# # <font color='red'>Observation:</font>
# ### <font color='white'>1. We observe that customers with less than 5 years of employment have a higher possibility to default on their loans. </font>
# ### <font color='white'>2. Customers with greater than 5 years of employment have less difficulties in paying their loans, hence they are a better group to attract on loan products. </font>

# ## **C) AMT_ANNUITY and AMT_CREDIT**

# In[127]:


fig,ax=plt.subplots(figsize=[10,8])

plt.rcParams.update({'font.size': 14})
sns.kdeplot(appl_data[appl_data.TARGET==0].AMT_ANNUITY, shade=True, color="green", label="on-time", ax=ax)
sns.kdeplot(appl_data[appl_data.TARGET==1].AMT_ANNUITY, shade=True, color="red", label="Late payments", ax=ax)
ax.set_ylabel("Density")
ax.set_xlabel("ANNUITY Amount")
fig.suptitle("Payment Status by Annuity Amount")
plt.show()


# In[128]:


fig,ax=plt.subplots(figsize=[10,8])

plt.rcParams.update({'font.size': 14})
sns.kdeplot(appl_data[appl_data.TARGET==0].AMT_CREDIT, shade=True, color="green", label="on-time", ax=ax)
sns.kdeplot(appl_data[appl_data.TARGET==1].AMT_CREDIT, shade=True, color="red", label="Late payments", ax=ax)
ax.set_ylabel("Density")
ax.set_xlabel("Credit Amount")
fig.suptitle("Payment Status by Credit Amount")
plt.show()


# # <font color='red'>Combined Observation for all the above kde plots:</font>
# ### <font color='white'>1. We observe that annuity amount below 20,000 have higher on-time payments rates. </font>
# ### <font color='white'>2. Customer with loan credits of 2.5 lakhs have a higher on-time payment than rest of the credit ranges. </font>
# ### <font color='white'>3. So we can conclude there is a low risk in giving out loans for 2.5lakhs or repayment annuity of below 20K </font>

# In[ ]:





# ### **7.2.2 Analysis between Numeric and Categorical variables**

# ### **A) Family Members**

# In[129]:


# Count of Childerns to the customer
plt.figure(figsize=[25,12])
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(appl_data.CNT_FAM_MEMBERS)
plt.xlabel("Family Members")
plt.ylabel("Count of Family Members")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.CNT_FAM_MEMBERS)
plt.xlabel("Family Members")
plt.ylabel("Count of Family Members")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# In[130]:


# Count of Childerns to the customer
plt.figure(figsize=[25,12])
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(on_time_appl_data.CNT_FAM_MEMBERS)
plt.xlabel("Family Members")
plt.ylabel("Count of Family Members")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.CNT_FAM_MEMBERS)
plt.xlabel("Family Members")
plt.ylabel("Count of Family Members")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# ### **B) Number of childrens**

# In[131]:


# Count of Childerns to the customer
plt.figure(figsize=[25,12])
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(on_time_appl_data.CNT_CHILDREN)
plt.xlabel("Count of Children")
plt.ylabel("Count of Applications")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.CNT_CHILDREN)
plt.xlabel("Count of Children")
plt.ylabel("Count of Applications")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# ### **C) Customer OWNs REALTY**

# In[132]:


plt.figure(figsize=[25,12])
plt.subplot(2,2,1)
plt.title("On-Time Payments")
sns.countplot(on_time_appl_data.FLAG_PHONE)
plt.xticks( rotation='horizontal')
plt.xlabel("OWN REALTY")
plt.ylabel("Count of People owning Realty")

plt.subplot(2,2,2)
plt.title("Late Payments")
sns.countplot(Late_Payments_appl_data.FLAG_PHONE)
plt.xticks( rotation='horizontal')
plt.xlabel("OWN REALTY")
plt.ylabel("Count of People owning Realty")
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# ### **D) Analysis based on Customer Age Group**

# In[133]:



plt.figure(figsize=[25,10])
plt.title("By Income category")
plt.subplot(2,2,1)
plt.title("AMT_GOODS_PRICE")
sns.barplot(y='AMT_GOODS_PRICE',x='AGE_GROUP_CUSTOMER',data = appl_data,hue = 'paystatus')
plt.subplot(2,2,2)
plt.title("AMT_ANNUITY")
sns.barplot(y='AMT_ANNUITY',x='AGE_GROUP_CUSTOMER',data = appl_data,hue = 'paystatus')
plt.subplot(2,2,3)

sns.barplot(y='AMT_INCOME_TOTAL',x='AGE_GROUP_CUSTOMER',data = appl_data,hue = 'paystatus')
plt.subplot(2,2,4)

sns.barplot(y='AMT_CREDIT',x='AGE_GROUP_CUSTOMER',data = appl_data,hue = 'paystatus')

plt.show()


# ## <font color='red'>Observation:</font> 
# ## <font color='white'> 1. We observe within age group 60-69 if their income total is more than 125K and credit more than 5Lakhs, then there is a difficulty to repay. So, we can say that if we offer them loans below 5Lakhs then there is a higher probability to repay installments on time. .</font>
# ## <font color='white'> 2. While considering annuity in range 20-25 thousand, all age groups apart from 60-69 have a higher possibility to default. </font>
# ## <font color='white'> 3. So we can conclude there is a low risk in giving out loans to age group 60-69 if the annuity is around 20-25 thousand and income is more than 1.25 Lakhs and credit amount is below 5 lakhs </font>

# In[ ]:





# 
# ### **E) Analysis based on Amount Income category**

# In[134]:


plt.figure(figsize=[30,14])
plt.suptitle("Wrong address given by Income category")
plt.subplot(2,2,1)
plt.title("REG_REGION_NOT_LIVE_REGION")
sns.barplot(y='REG_REGION_NOT_LIVE_REGION',x='AMT_INCOME_CATEG',data = appl_data,hue = 'paystatus')
plt.subplot(2,2,2)
plt.title("REG_CITY_NOT_LIVE_CITY")
sns.barplot(y='REG_CITY_NOT_LIVE_CITY',x='AMT_INCOME_CATEG',data = appl_data,hue = 'paystatus')
plt.subplot(2,2,3)
plt.title("REG CITY NOT WORK CITY")
sns.barplot(x='AMT_INCOME_CATEG',y='REG_CITY_NOT_WORK_CITY',data = appl_data,hue = 'paystatus')
plt.subplot(2,2,4)
plt.title("LIVE CITY NOT WORK CITY")
sns.barplot(x='AMT_INCOME_CATEG',y='LIVE_CITY_NOT_WORK_CITY',data = appl_data,hue = 'paystatus')
plt.show()


# ## <font color='red'>Observation:</font> 
# ## <font color='white'> 1. We observed that no matter what income category the customer belongs to, if they provide wrong contact address then they have a higher probability to default. As all the bars indicate higher late payment values when address is not provided .</font>

# ### **F) Analysis based on Family Status and Goods price**

# In[135]:


plt.figure(figsize=[10,7])
plt.rcParams.update({'font.size': 12})
plt.title("By Family Status and Goods price")
sns.barplot(x='NAME_FAMILY_STATUS',y='AMT_CREDIT',data = appl_data,hue = 'paystatus')


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# ### **G) Analysis based on Age group and Family/Children counts**

# In[136]:


plt.figure(figsize=[25,10])
plt.subplot(2,2,1)
plt.title("By Age Group and Family counts")
sns.barplot(x='AGE_GROUP_CUSTOMER',y='CNT_FAM_MEMBERS',data = appl_data,hue = 'paystatus')

plt.subplot(2,2,2)
plt.title("By Age Group and Children count")
sns.barplot(x='AGE_GROUP_CUSTOMER',y='CNT_CHILDREN',data = appl_data,hue = 'paystatus')
plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>No key observation to report, only counts are different</font>

# ### **7.2.3 Multi variate Analysis**

# In[137]:


#find the correlation between application data attributes

plt.figure(figsize=[25,15])
app_data_Corr= appl_data.corr().sort_values(by="TARGET",ascending=False)
app_data_Corr=app_data_Corr.apply(lambda x: round(abs(x),2))
sns.heatmap(app_data_Corr,annot=True)
plt.savefig('heatmap.png')
plt.show()


# ### Based on the above heatmap, we have identified the highly corelated attributes: 

# In[138]:


app_unstack = app_data_Corr.unstack()
list_of_corr = app_unstack.sort_values()
list_of_corr[(list_of_corr>.60) & (list_of_corr<1.0)].sort_values(ascending=False)


# # <font color='Green'>**8. Exploring the Data in Previous Application Data file</font>** 

# ## **8.1 Exploring the Data**

# In[139]:


p_appl_data.head()


# In[140]:


p_appl_data.info()


# In[141]:


p_appl_data.shape


# In[142]:


p_appl_data.describe()


# ## **8.2 Checking percentage of Null values Present in the Data set**

# In[143]:


null_cols=((p_appl_data.isnull().sum()*100)/p_appl_data.shape[0]).round(2)
null_cols[null_cols>0]


# ## **8.3 Drop columns which have more than 20% of Null values in them**

# In[144]:


p_appl_data.loc[:,p_appl_data.isnull().mean()>=.20]


# In[145]:


p_appl_data.drop(p_appl_data.loc[:,p_appl_data.isnull().mean()>=.20],axis=1,inplace=True)


# In[146]:


((p_appl_data.isnull().sum()*100)/p_appl_data.shape[0]).round(2)


# In[147]:


p_appl_data.shape


# ## **8.4 Impute the Null values with mean, median, mode or 0  where applicable**

# In[148]:


p_appl_data.PRODUCT_COMBINATION.mode()


# In[149]:


p_appl_data['PRODUCT_COMBINATION'].fillna(p_appl_data.PRODUCT_COMBINATION.mode()[0], inplace=True)
p_appl_data.PRODUCT_COMBINATION.isnull().sum()


# ###### Are there any objects that needs to be converted into numbers ? Looks good!

# In[150]:


p_appl_data.select_dtypes('object')


# In[ ]:





# ## **8.5 Fixing Outliers**

# In[151]:


p_appl_data.info()


# In[152]:


list_datatype=['float64','int64']
data = list(p_appl_data.select_dtypes(include=list_datatype).columns)
data = data[2:]
data


# In[153]:


for cols in data:
    f, ax = plt.subplots(1,1,figsize=(10,2))
    f.suptitle(cols, fontsize=16)
    sns.boxplot(p_appl_data[cols],color='green')
    plt.show()


# ### **A) AMT_APPLICATION**

# In[154]:


p_appl_data=p_appl_data[p_appl_data.AMT_APPLICATION < p_appl_data.AMT_APPLICATION.quantile(0.75)]


# In[155]:


sns.boxplot(p_appl_data.AMT_APPLICATION,color='green')


# In[156]:


p_appl_data=p_appl_data[p_appl_data.AMT_CREDIT < p_appl_data.AMT_CREDIT.quantile(0.90)]


# ### **B) AMT_CREDIT**

# In[157]:


sns.boxplot(p_appl_data.AMT_CREDIT,color='green')


# In[ ]:





# In[ ]:





# ## **Overall Decisions on previous applications**

# In[158]:


plt.figure(figsize=[25,14])
plt.subplot(2,2,1)
plt.title("By Amount Application")
sns.barplot(x = 'NAME_CONTRACT_STATUS', y="AMT_APPLICATION",data=p_appl_data)
plt.subplot(2,2,2)
plt.title("By Amount credit")
sns.barplot(x = 'NAME_CONTRACT_STATUS', y="AMT_CREDIT",data=p_appl_data)
plt.show()


# ### Relationshiop between application amount and credit amount

# In[159]:


sns.scatterplot(x = 'AMT_APPLICATION', y="AMT_CREDIT",data=p_appl_data,color='green')
plt.show()


# #### Ideally credit amount should not be greater than application amount. So removing those data anomolies

# In[160]:


p_appl_data=p_appl_data[~(p_appl_data.AMT_CREDIT > p_appl_data.AMT_APPLICATION)]


# In[161]:


sns.scatterplot(x = 'AMT_APPLICATION', y="AMT_CREDIT",data=p_appl_data,color='green')
plt.show()


# ## **8.6 Analyzing data graphically and idenifying key insights and correlations**

# In[162]:


p_appl_data


# In[163]:


Filter_data = p_appl_data[(p_appl_data.NAME_CASH_LOAN_PURPOSE != 'XAP') & (p_appl_data.NAME_CASH_LOAN_PURPOSE != 'XNA') & (p_appl_data.NAME_CASH_LOAN_PURPOSE != 'Other')]
plt.figure(figsize=[10,2])
plt.xticks( rotation='vertical')
sns.countplot(Filter_data[Filter_data.NAME_CONTRACT_TYPE=='Cash loans'].NAME_CASH_LOAN_PURPOSE)

plt.show()


# ## <font color='red'>Observation:</font> <font color='white'>Most number of cash loan applications are for Repairs. </font>

# In[164]:


data = list(p_appl_data.NAME_CONTRACT_STATUS.value_counts().index)
data


# In[165]:


plt.figure(figsize=[25,10])
plt.title("Cash loan Status by Credit Amount")
plt.xticks( rotation='vertical')
sns.barplot(y='AMT_CREDIT',x='NAME_CASH_LOAN_PURPOSE',data = Filter_data[Filter_data.NAME_CONTRACT_TYPE=='Cash loans'],hue = 'NAME_CONTRACT_STATUS',palette='Set1')
plt.show()


# ## <font color='red'>Observation:</font> 
# ## <font color='white'>1. Buying a home has the most cash loan approvals in terms of credit amounts </font>
# ## <font color='white'>2. Journey and building a house has the most number of cash loan cancellations</font>
# ## <font color='white'>3. Business developement has the highest ratio of credit amounts approved vs refused.</font>
# ## <font color='white'>4. Hobby, Gasification,buying new car,buying a garage has no cancellations</font>
# 
# 

# In[ ]:





# In[166]:


fig,ax=plt.subplots(figsize=[10,8])

plt.rcParams.update({'font.size': 14})
sns.kdeplot(p_appl_data[p_appl_data.NAME_CONTRACT_STATUS == data[0]].groupby(by=['NAME_CONTRACT_TYPE','NAME_CONTRACT_STATUS']).SK_ID_PREV.count(), shade=True, color="green", label="approved", ax=ax)
sns.kdeplot(p_appl_data[p_appl_data.NAME_CONTRACT_STATUS == data[2]].groupby(by=['NAME_CONTRACT_TYPE','NAME_CONTRACT_STATUS']).SK_ID_PREV.count(), shade=True, color="red", label="refused", ax=ax)
sns.kdeplot(p_appl_data[p_appl_data.NAME_CONTRACT_STATUS == data[1]].groupby(by=['NAME_CONTRACT_TYPE','NAME_CONTRACT_STATUS']).SK_ID_PREV.count(), shade=True, color="orange", label="Cancelled", ax=ax)
sns.kdeplot(p_appl_data[p_appl_data.NAME_CONTRACT_STATUS == data[3]].groupby(by=['NAME_CONTRACT_TYPE','NAME_CONTRACT_STATUS']).SK_ID_PREV.count(), shade=True, color="gray", label="Unused", ax=ax)

ax.set_ylabel("Density")
ax.set_xlabel("Credit Amount")
fig.suptitle("Payment Status by Credit Amount")
plt.show()


# ## Contract Type

# In[167]:


plt.figure(figsize=[20,8])
plt.title("Credit Amounts by Contract Type")
sns.barplot(y='AMT_CREDIT',x='NAME_CONTRACT_TYPE',data = p_appl_data[p_appl_data.NAME_CONTRACT_TYPE!='XNA'],hue = 'NAME_CONTRACT_STATUS',palette='Set1')
plt.show()


# ## <font color='red'>Observation:</font> 
# ## <font color='white'>1. Cash loans are approved for higher credit amounts than Consumer and revolving loans and refused the least.  </font>
# ## <font color='white'>2. Consumer loans have higher cancellations as compated to cash loans and revolving loans</font>

# In[ ]:





# ## Client Type

# In[168]:


plt.figure(figsize=[20,8])
plt.title("Credit Amounts by Client Type")
sns.barplot(y='AMT_CREDIT',x='NAME_CLIENT_TYPE',data = p_appl_data[p_appl_data.NAME_CLIENT_TYPE!='XNA'],hue = 'NAME_CONTRACT_STATUS',palette='Set1')
plt.show()


# ## <font color='red'>Observation:</font> 
# ## <font color='white'>1. Loans for greater than 70K+ are getting approved for Repeaters and Refreshed client as compared to New clients.  </font>
# ## <font color='white'>2. New clients have higher cancellations as compated to repeaters and refreshed clients</font>
# ## <font color='white'>3. New clients are refused on high credit loan applications. </font>

# ## **Loan Status VS Product Combinations**

# In[169]:


res = pd.pivot_table(data=p_appl_data, index="PRODUCT_COMBINATION", columns="NAME_CONTRACT_STATUS", values="AMT_APPLICATION",aggfunc='mean')
res = res[['Approved','Canceled','Refused','Unused offer']].apply(lambda x: round(x/100000,2))
sns.heatmap(res, cmap="RdYlGn", annot= True,  center=0.427)

plt.show()


# ## **Loan Application Amount vs Loan Received Amount**

# In[170]:


plt.figure(figsize=[25,14])
plt.suptitle("Application Amount Vs Credit Amount")
plt.subplot(2,2,1)
plt.title("Approved")
sns.scatterplot(x = 'AMT_APPLICATION', y="AMT_CREDIT",data=p_appl_data[p_appl_data.NAME_CONTRACT_STATUS == data[0]],color='green')
plt.subplot(2,2,2)
plt.title("Canceled")
sns.scatterplot(x = 'AMT_APPLICATION', y="AMT_CREDIT",data=p_appl_data[p_appl_data.NAME_CONTRACT_STATUS == data[1]],color='orange')
plt.subplot(2,2,3)
plt.title("Refused")
sns.scatterplot(x = 'AMT_APPLICATION', y="AMT_CREDIT",data=p_appl_data[p_appl_data.NAME_CONTRACT_STATUS == data[2]],color='red')
plt.subplot(2,2,4)
plt.title("Unused Offer")
sns.scatterplot(x = 'AMT_APPLICATION', y="AMT_CREDIT",data=p_appl_data[p_appl_data.NAME_CONTRACT_STATUS == data[3]],color='gray')
plt.show()


# In[ ]:





# # <font color='red'>Observation:</font>
# ## <font color='white'>1. We can see a linear progression between credit amount and application amount</font>
# ## <font color='white'>2. We observed on cancelled applications the amount credit and amount application differs by some degree. This can be a reason why applications may have been cancelled</font>
# 

# In[ ]:





# # <font color='Green'>**9. Data Merging of Application data </font>** 

# In[171]:


combined_data = pd.merge(appl_data, p_appl_data, how='left', on=['SK_ID_CURR'])
combined_data


# In[172]:


combined_data.shape


# ## **Combined Correlation of all factoring attributes**

# In[173]:


combined_data_corr= combined_data.corr().sort_values(by="TARGET",ascending=False)
plt.figure(figsize=[50,25])
combined_data_corr=combined_data_corr.apply(lambda x : round(abs(x),2))
sns.heatmap(combined_data_corr,annot=True)
plt.show()


# In[174]:


p_app_unstack = combined_data_corr.unstack()
list_of_corr = p_app_unstack.sort_values(kind="quicksort")
list_of_corr[(list_of_corr>.60) & (list_of_corr<1.0)].sort_values(ascending=False)


# In[ ]:





# ## **Decisions based analysis on Cobmined data**

# ## **Loan Decisions and Pay status based on Channel Type**

# In[176]:


plt.figure(figsize=[25,12])

plt.suptitle("Loan Decisions based on Channel Type")
plt.subplot(2,2,1)
plt.title("Amount Application")
plt.xticks( rotation='vertical')
sns.barplot(y = 'AMT_APPLICATION', x="CHANNEL_TYPE",data=p_appl_data,hue = "NAME_CONTRACT_STATUS" )
plt.subplot(2,2,2)
plt.title("Amount Credit")
plt.xticks( rotation='vertical')
sns.barplot(y = 'AMT_CREDIT_y', x="CHANNEL_TYPE",data=combined_data,hue = "paystatus")

plt.show()



# # <font color='red'>Observation:</font>
# ## <font color='white'>1. Car dealers are good customers as they have less late payments and are approved more than they are refused.</font>
# ## <font color='white'>2. Contact center are highly approved and have almost equal on-time vs late payment ratio.</font>
# ## <font color='white'>3. Channel of corp sales are risky as they have higher late payment ratios and are refused loans more than approved.</font>
# 

# In[ ]:




