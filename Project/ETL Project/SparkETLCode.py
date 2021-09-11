#!/usr/bin/env python
# coding: utf-8

# # **Student_ Name: Pappu Dindayal Kapgate**

# In[1]:


# Setting Up the environment

import os
import sys
os.environ["PYSPARK_PYTHON"] = "/opt/cloudera/parcels/Anaconda/bin/python"
os.environ["JAVA_HOME"] = "/usr/java/jdk1.8.0_232-cloudera/jre"
os.environ["SPARK_HOME"]="/opt/cloudera/parcels/SPARK2-2.3.0.cloudera2-1.cdh5.13.3.p0.316101/lib/spark2/"
os.environ["PYLIB"] = os.environ["SPARK_HOME"] + "/python/lib"
sys.path.insert(0, os.environ["PYLIB"] +"/py4j-0.10.6-src.zip")
sys.path.insert(0, os.environ["PYLIB"] +"/pyspark.zip")


# # **1. Creating Spark APP**

# In[2]:


# importing necessarry lib
from pyspark.sql import SparkSession


# In[3]:


# creating spark app

spark = SparkSession.builder.appName('atm_trans_data').master("local").getOrCreate()


# In[4]:


# exploring app information
spark


# # **2. Importing data**

# In[5]:


# setting up the data path
hdfs_input_data_path = '/user/root/atm_tran/part-m-00000'


# In[6]:


# Reading data into the datafram
atm_data = spark.read.format("csv").option("header", "True").load(hdfs_input_data_path)


# In[7]:


# checking the schema
atm_data.printSchema()


# In[8]:


# Checking the count
atm_data.count()


# **Reading data and assinging the userdefine schema file.Due to the header is not present in above datafram and the data type is not correct.**

# In[9]:


# Importing necessary data type lib
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType,DoubleType


# In[10]:


# Creating the Schema

fileSchema = StructType([StructField('year', IntegerType(),True),
                        StructField('month', StringType(),True),
                        StructField('day', IntegerType(),True),
                        StructField('weekday', StringType(),True),
                        StructField('hour', IntegerType(),True),
                        StructField('atm_status', StringType(),True),
                        StructField('atm_id', StringType(),True),
                        StructField('atm_manufacturer', StringType(),True),
                        StructField('atm_location', StringType(),True),
                        StructField('atm_streetname', StringType(),True),
                        StructField('atm_street_number', IntegerType(),True),
                        StructField('atm_zipcode', IntegerType(),True),
                        StructField('atm_lat', DoubleType(),True),
                        StructField('atm_lon', DoubleType(),True),
                        StructField('currency', StringType(),True),
                        StructField('card_type', StringType(),True),
                        StructField('transaction_amount', IntegerType(),True), 
                        StructField('service', StringType(),True), 
                        StructField('message_code', StringType(),True),
                        StructField('message_text', StringType(),True),
                        StructField('weather_lat', DoubleType(),True),
                        StructField('weather_lon', DoubleType(),True),
                        StructField('weather_city_id ', IntegerType(),True),
                        StructField('weather_city_name', StringType(),True),
                        StructField('temp', DoubleType(),True), 
                        StructField('pressure', IntegerType(),True),
                        StructField('humidity', IntegerType(),True),
                        StructField('wind_speed', IntegerType(),True),
                        StructField('wind_deg', IntegerType(),True),
                        StructField('rain_3h', DoubleType(),True),
                        StructField('clouds_all', IntegerType(),True),
                        StructField('weather_id', IntegerType(),True), 
                        StructField('weather_main', StringType(),True),
                        StructField('weather_description', StringType(),True),     
                        ])


# In[11]:


# Reading data into datafram with schema
atm_data = spark.read.load("/user/root/atm_tran/part-m-00000",format="csv", sep=",", schema=fileSchema, header=False)


# In[12]:


# Checking the schema
atm_data.printSchema()


# In[13]:


#Checking the count
atm_data.count()


# In[14]:


# Creating the view of "atm_data" datafrma to execute sql query 
atm_data.createOrReplaceTempView("atm_sql_data")


# # **3. Fact and Dim Table**

# ## 3.1 DIM_LOCATION

# In[15]:


# importing lib 
from pyspark.sql.functions import row_number,lit
from pyspark.sql.window import Window
# creating the windows for row_number(to genrate index id/number)
w = Window().orderBy(lit('A'))


# In[16]:


# Creating dimention table of location as DIM_LOCATION
DIM_LOCATION_TABLE = spark.sql("select distinct atm_location as location, atm_streetname as streetname,                                atm_street_number as street_number, atm_zipcode as zipcode, atm_lat as lat,                                atm_lon as lon from atm_sql_data").withColumn("location_id", row_number().over(w))


# In[17]:


# Rearrangment of columns according to the target schema

dim_loc_col_name = DIM_LOCATION_TABLE.columns                    #Extracting the column name and storing in list
dim_loc_col_name = dim_loc_col_name[-1:] + dim_loc_col_name[:-1] #Rearanging the column names in the list 
DIM_LOCATION_TABLE = DIM_LOCATION_TABLE.select(dim_loc_col_name) #Extracting data as target schema


# In[18]:


# Checking the Schema
DIM_LOCATION_TABLE.printSchema()


# In[19]:


# Checking the count
DIM_LOCATION_TABLE.count()


# In[20]:


# Printing the data from the dataframw
DIM_LOCATION_TABLE.show(5)


# In[21]:


# Creating the view for the sql query
DIM_LOCATION_TABLE.createOrReplaceTempView("atm_location_sql_table")


# In[22]:


# Data writing path to s3 bucket 
output_path = "s3a://atmtransactiondata/dim_location"


# In[23]:


# Data writing path to s3 bucket 
output_path = "s3a://atmtransactiondata/dim_location"
# Writing the data into s3 bucket
DIM_LOCATION_TABLE.write.csv(output_path, mode = 'overwrite')


# ## 3.2 DIM_ATM

# In[24]:


# Creating dim atm table datafram
DIM_ATM_TABLE = spark.sql("select distinct atm_id as atm_number , atm_manufacturer, AL.location_id as atm_location_id                     from atm_sql_data A  left outer join  atm_location_sql_table AL                     on                     A.atm_lon = AL.lon                     and                     A.atm_lat = AL.lat").withColumn("atm_id", row_number().over(w))


# In[25]:


# Rearrangment of columns according to the target schema

dim_atm_col_name = DIM_ATM_TABLE.columns                    #Extracting the column name and storing in list
dim_atm_col_name = dim_atm_col_name[-1:] + dim_atm_col_name[:-1] #Rearanging the column names in the list 
DIM_ATM_TABLE = DIM_ATM_TABLE.select(dim_atm_col_name) #Extracting data as target schema


# In[26]:


#Checking the Schema
DIM_ATM_TABLE.printSchema()


# In[27]:


# Checking the count
DIM_ATM_TABLE.count()


# In[28]:


# Display the data
DIM_ATM_TABLE.show()


# In[29]:


# Creating the view for sql query
DIM_ATM_TABLE.createOrReplaceTempView("dim_atm_sql_table")


# In[30]:


# Data writing path to s3 bucket 
output_path = "s3a://atmtransactiondata/dim_atm"
# Writing the data into s3 bucket
DIM_ATM_TABLE.write.csv(output_path, mode = 'overwrite')


# ## 3.3 DIM_DATE

# In[31]:


# Importing necesarry lib
from pyspark.sql.functions import *
from pyspark.sql.functions import row_number,lit
from pyspark.sql.window import Window
w = Window().orderBy(lit('A'))

# converting month column text in to the numeric month column
# after convertion adding a column into the main datafram
atm_data = atm_data.withColumn("mnth_name",from_unixtime(unix_timestamp(atm_data.month,'MMM'),'MM'))

# after adding month column into the main datafram creating the date_column into the main datafram
atm_data = atm_data.withColumn('date_column',concat_ws('-', atm_data.year, atm_data.mnth_name, atm_data.day))

# date format
pattern1 = 'yyyy-MM-dd'

# converting the date column into the specific data type format
atm_data = atm_data.withColumn('date_column', to_date(unix_timestamp(atm_data['date_column'], pattern1).cast('timestamp')))

#Display the date column data
atm_data.select('date_column').show(5)


# In[32]:


# Creating column that concate the hours with date column
atm_data = atm_data.withColumn('full_date_time',concat_ws('-',atm_data.date_column, atm_data.hour))
atm_data.select('full_date_time').show(5)


# In[33]:


# date and time format
pattern2 = 'yyyy-MM-dd-HH'

# Creating date and time column
atm_data = atm_data.withColumn("full_date_time", from_unixtime(unix_timestamp(atm_data['full_date_time'],pattern2)).cast("timestamp")) 

# Display the data
atm_data.select('full_date_time').show(5)


# In[34]:


# Recreating the view of atm_data framwork
atm_data.createOrReplaceTempView("atm_sql_data")


# In[35]:


# Creating the dim_date dimention table
DIM_DATE_TABLE = spark.sql("select distinct full_date_time, year, month, day, hour, weekday                      from atm_sql_data").withColumn("date_id", row_number().over(w))


# In[36]:


# Rearrangment of columns according to the target schema

dim_date_col_name = DIM_DATE_TABLE.columns                    #Extracting the column name and storing in list
dim_date_col_name = dim_date_col_name[-1:] + dim_date_col_name[:-1] #Rearanging the column names in the list 
DIM_DATE_TABLE = DIM_DATE_TABLE.select(dim_date_col_name) #Extracting data as target schema


# In[37]:


# Checking the Schema
DIM_DATE_TABLE.printSchema()


# In[38]:


# Checking the count
DIM_DATE_TABLE.count()


# In[39]:


# Displaying the data
DIM_DATE_TABLE.show(20)


# In[40]:


# Creating the view for sql query
DIM_DATE_TABLE.createOrReplaceTempView("dim_date_sql_table")


# In[41]:


# Data writing path to s3 bucket 
output_path = "s3a://atmtransactiondata/dim_date"
# Writing the data into s3 bucket
DIM_DATE_TABLE.write.csv(output_path, mode = 'overwrite')


# ## 3.4 DIM_CARD_TYPE

# In[42]:


# Creating card type dimention table
DIM_CARD_TYPE_TABLE = spark.sql("select distinct card_type from atm_sql_data").withColumn("card_type_id", row_number().over(w))


# In[43]:


# Rearrangment of columns according to the target schema

dim_card_type_name = DIM_CARD_TYPE_TABLE.columns                    #Extracting the column name and storing in list
dim_card_type_name = dim_card_type_name[-1:] + dim_card_type_name[:-1] #Rearanging the column names in the list 
DIM_CARD_TYPE_TABLE = DIM_CARD_TYPE_TABLE.select(dim_card_type_name) #Extracting data as target schema


# In[44]:


# Checking the schema of dataframe

DIM_CARD_TYPE_TABLE.printSchema()


# In[45]:


# Checking the count

DIM_CARD_TYPE_TABLE.count()


# In[46]:


# Display the data
DIM_CARD_TYPE_TABLE.show()


# In[47]:


# Creating the view for sql query
DIM_CARD_TYPE_TABLE.createOrReplaceTempView("dim_card_type_sql_table")


# In[48]:


# Data writing path to s3 bucket 
output_path = "s3a://atmtransactiondata/dim_card_type"
# Writing the data into s3 bucket
DIM_CARD_TYPE_TABLE.write.csv(output_path, mode = 'overwrite')


# ## 3.5 FACT_ATM_TRANS

# In[49]:


# Creating the fact table by using dim datafram and main(atm_data) datafram 
FACT_ATM_TRANS_TABLE = atm_data.join(DIM_CARD_TYPE_TABLE,
                                    (atm_data.card_type == DIM_CARD_TYPE_TABLE.card_type),'outer').join(
    DIM_LOCATION_TABLE,(atm_data.atm_lat == DIM_LOCATION_TABLE.lat) & 
    (atm_data.atm_lon == DIM_LOCATION_TABLE.lon) & 
    (atm_data.atm_location == DIM_LOCATION_TABLE.location) & 
    (atm_data.atm_zipcode == DIM_LOCATION_TABLE.zipcode) & 
    (atm_data.atm_streetname == DIM_LOCATION_TABLE.streetname) & 
    (atm_data.atm_street_number == DIM_LOCATION_TABLE.street_number),'outer').join(
    DIM_DATE_TABLE,(atm_data.year == DIM_DATE_TABLE.year) & 
    (atm_data.month == DIM_DATE_TABLE.month) & 
    (atm_data.day == DIM_DATE_TABLE.day) & 
    (atm_data.hour == DIM_DATE_TABLE.hour) & 
    (atm_data.weekday == DIM_DATE_TABLE.weekday),'outer').select(
    DIM_CARD_TYPE_TABLE.card_type_id,
    atm_data.transaction_amount,
    atm_data.service,
    atm_data.currency,
    atm_data.atm_status,
    atm_data.message_code,
    atm_data.message_text,
    atm_data.rain_3h,
    atm_data.clouds_all,
    atm_data.weather_id,
    atm_data.weather_main,
    atm_data.weather_description,
    col("location_id").alias("weather_loc_id"),
    atm_data.atm_manufacturer,
    atm_data.atm_id,
    DIM_DATE_TABLE.date_id)


# In[50]:


# Joing the atm dimention table with fact table
FACT_ATM_TRANS_TABLE = FACT_ATM_TRANS_TABLE.join(DIM_ATM_TABLE,
                                 (FACT_ATM_TRANS_TABLE.weather_loc_id == DIM_ATM_TABLE.atm_location_id) & 
                                 (FACT_ATM_TRANS_TABLE.atm_manufacturer == DIM_ATM_TABLE.atm_manufacturer) & 
                                 (FACT_ATM_TRANS_TABLE.atm_id == DIM_ATM_TABLE.atm_number),'inner').select(
    DIM_ATM_TABLE.atm_id,
    FACT_ATM_TRANS_TABLE.weather_loc_id,
    FACT_ATM_TRANS_TABLE.date_id,
    FACT_ATM_TRANS_TABLE.card_type_id,
    FACT_ATM_TRANS_TABLE.atm_status,
    FACT_ATM_TRANS_TABLE.currency,
    FACT_ATM_TRANS_TABLE.service,
    FACT_ATM_TRANS_TABLE.transaction_amount,
    FACT_ATM_TRANS_TABLE.message_code,
    FACT_ATM_TRANS_TABLE.message_text,
    FACT_ATM_TRANS_TABLE.rain_3h,
    FACT_ATM_TRANS_TABLE.clouds_all,
    FACT_ATM_TRANS_TABLE.weather_id,
    FACT_ATM_TRANS_TABLE.weather_main,
    FACT_ATM_TRANS_TABLE.weather_description)


# In[51]:


FACT_ATM_TRANS_TABLE = FACT_ATM_TRANS_TABLE.withColumn('trans_id', row_number().over(w))


# In[52]:


# Rearrangment of columns according to the target schema

dim_fact_trans_name = FACT_ATM_TRANS_TABLE.columns                    #Extracting the column name and storing in list
dim_fact_trans_name = dim_fact_trans_name[-1:] + dim_fact_trans_name[:-1] #Rearanging the column names in the list 
FACT_ATM_TRANS_TABLE = FACT_ATM_TRANS_TABLE.select(dim_fact_trans_name) #Extracting data as target schema


# In[53]:


# Checking the Schema
FACT_ATM_TRANS_TABLE.printSchema()


# In[54]:


# Display the data

FACT_ATM_TRANS_TABLE.show()


# In[55]:


# Checking the count of datafram
FACT_ATM_TRANS_TABLE.count()


# In[56]:


# Data writing path to s3 bucket 
output_path = "s3a://atmtransactiondata/fact_atm_trans"
# Writing the data into s3 bucket
FACT_ATM_TRANS_TABLE.write.csv(output_path, mode = 'overwrite')

