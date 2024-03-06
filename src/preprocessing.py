#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import missingno
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data/accepted_2007_to_2018Q4.csv')

#Loan_Status is the target column.
print('TARGET FEATURE: loan_status')
print(df['loan_status'].value_counts())

#Since the model is about predicting whether a certain loan will be repayed or not
#Therefore we only need cases where the loan is
# - Fully paid
# - Charged off (not repayed)
df = df[df['loan_status'].isin(['Fully Paid','Charged Off'])]
df


# In[2]:


print(f'The data has {df.shape[0]} ROWS and {df.shape[1]} COLUMNS before processing')


# In[3]:


#there are two types of applications, individual and joint application.
#Some of the columns are specific to joint type applications
#filling the rest with some value like -1 allows us to keep them but it affects the data
#so, we separate them and work on them separately.
df['application_type'].value_counts()


# In[4]:


df_ind = df[df['application_type'] == 'Individual']
print('Individual application count:',df_ind.shape[0])
df_joint = df[df['application_type'] == 'Joint App']
print('Joint application count:',df_joint.shape[0])


# # Application_type: Individual
# ## Preparing the data

# In[5]:


df_ind['loan_status'].value_counts()


# #### i. Dropping  with more than 30% missing values

# In[6]:


#missing value using missingno


missingno.matrix(df_ind)
plt.title('Before dropping missing values', size = 25)
plt.show()
plt.savefig('figs/before_missing_value_filter.png')


# In[7]:


#finding columns with more than 30% null values.
print('Gathering columns with more than 30% null values')
threshold = 0.3 * df_ind.shape[0]
drop_cols_ind = []
for i in df_ind.columns:
    if(df_ind[i].isna().sum() > threshold):
        print(f'- {i} has {df_ind[i].isna().sum() / df_ind.shape[0] * 100}% missing values')
        #df.drop([i], axis=1, inplace=True)
        drop_cols_ind.append(i)
print(f'Each of {len(drop_cols_ind)} above have more than 30% missing values ')
#print(f'The data has {df.shape[0]} ROWS and {df.shape[1]} COLUMNS after dropping columns having more than 30% missing values')


# In[8]:


#Dropping
ind_df = df_ind.drop(drop_cols_ind, axis=1).copy()
print(f'The number of individual applicant record data have {ind_df.shape[0]} ROWS and {ind_df.shape[1]} COLUMNS')


# #### ii. Dropping rows with missing values

# In[9]:


#we can see a chunk of data missing if few of the columns. Lets set a threshold to drop those rows too.
#Rows are allows to have 20 missing values out of 93. If there are more than that, drop those rows.
print(f'Before : {ind_df.shape[0]} rows X {ind_df.shape[1]} columns')
ind_df = ind_df.dropna(thresh=73)
print(f'After : {ind_df.shape[0]} rows X {ind_df.shape[1]} columns')

missingno.matrix(ind_df)
plt.title('After dropping missing values', size = 25)
plt.show()
plt.savefig('figs/after_missing_value_filter.png')


# #### iii. Missing value report for the rest of the columns

# In[10]:


#Missing value report in percentages
mvr = ind_df.isna().sum() / ind_df.shape[0] *100
for i, j in mvr.items():
    print(f'{i} - {"{:.10f}".format(j)}')


# #### iii. Preprocessing other columns

# In[11]:


#id column is unique
if(len(ind_df['id'].unique()) == ind_df.shape[0]):
    print(f'- "id" is unique id column - making it index')

ind_df.set_index(['id'], inplace=True)

ind_df['term'] = ind_df['term'].str.strip()
#term is specified as months only, therefore create a new column 
ind_df['term_months'] = ind_df['term'].str.split(' ').str[0]
ind_df['term_months'] = ind_df['term_months'].astype('float')
ind_df.drop(['term'], axis=1, inplace=True)
print('- Created "term_months" and dropped "term"')

#changing emp_length to integer col in terms of years

ind_df.loc[ind_df['emp_length'] == '< 1 year','emp_length'] = '0+ years'
ind_df['emp_length'] = ind_df['emp_length'].str.extract('(\d+)').astype('float')
print('- Converted "emp_length" to float col')

#dropping emp_title as there are so many categories and its importance cannot be understood
ind_df.drop(['emp_title'], axis=1, inplace=True)
print('- Dropped "emp_title"')

#Dropping since its a date column.
#ind_df['issue_month'] = ind_df['issue_d'].str.split('-').str[0]
#ind_df['issue_year'] = ind_df['issue_d'].str.split('-').str[1]
ind_df.drop(['issue_d'], axis=1, inplace=True)
print('- Dropped "issue_d"')

#Dropping since its a date column.
#ind_df['earliest_cr_line_month'] = ind_df['earliest_cr_line'].str.split('-').str[0]
#ind_df['earliest_cr_line_year'] = ind_df['earliest_cr_line'].str.split('-').str[1]
ind_df.drop(['earliest_cr_line'], axis=1, inplace=True)
print('- Dropped "earliest_cr_line"')

#payment plan mostly "no" value
ind_df.drop(['pymnt_plan'], axis=1, inplace=True)
print('- Dropped "pymnt_plan"')

#dropping url column
ind_df.drop(['url'], axis=1, inplace=True)
print('- Dropped "url"')

#dropping title as it might not be important
ind_df.drop(['title'], axis=1, inplace=True)
print('- Dropped "title"')

#dropping zipcode as it is incomplete and might not carry enough information
ind_df.drop(['zip_code'], axis=1, inplace=True)
print('- Dropped "zip_code"')

#dropping out_prncp as all values are the same - 0
ind_df.drop(['out_prncp'], axis=1, inplace=True)
print('- Dropped "out_prncp"')

#dropping out_prncp_inv as all values are the same - 0
ind_df.drop(['out_prncp_inv'], axis=1, inplace=True)
print('- Dropped "out_prncp_inv"')

#dropping last_pymnt_d
#ind_df['last_pymnt_month'] = ind_df['last_pymnt_d'].str.split('-').str[0]
#ind_df['last_pymnt_year'] = ind_df['last_pymnt_d'].str.split('-').str[1]
ind_df.drop(['last_pymnt_d'], axis=1, inplace=True)
print('- Dropped "last_pymnt_d"')

#Dropping last_credit_pull_d
#ind_df['last_credit_pull_month'] = ind_df['last_credit_pull_d'].str.split('-').str[0]
#ind_df['last_credit_pull_year'] = ind_df['last_credit_pull_d'].str.split('-').str[1]
ind_df.drop(['last_credit_pull_d'], axis=1, inplace=True)
print('- Dropped "last_credit_pull_d"')

#dropping policy_code as all values are the same - 1
ind_df.drop(['policy_code'], axis=1, inplace=True)
print('- Dropped "policy_code"')

#dropping application_type as all values are the same - individual
ind_df.drop(['application_type'], axis=1, inplace=True)
print('- Dropped "application_type"')

#dropping hardship_flag as all values are the same - N
ind_df.drop(['hardship_flag'], axis=1, inplace=True)
print('- Dropped "hardship_flag"')


# In[12]:


#more columns to be dropped
drop_cols = ind_df.filter(regex=r'^last_').columns.tolist()
drop_cols.extend(['collection_recovery_fee','recoveries', 'total_pymnt', 'total_pymnt_inv', 
                  'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp',
                 'total_pymnt','total_pymnt_inv','total_rec_prncp',
                 'total_rec_int','total_rec_late_fee','num_tl_120dpd_2m',
                 'num_tl_30dpd','debt_settlement_flag'])

print('Dropping:')
for i in drop_cols:
    print(f'- {i}')


# In[13]:


ind_df.drop(drop_cols, axis = 1, inplace = True)


# In[14]:


ind_df


# In[15]:


print(f'The dataframe has {ind_df.shape[0]} rows and {ind_df.shape[1]} columns')


# In[16]:


print('Saving the data into new file...')
ind_df.to_csv('data/individual_application_information.csv')
print('Saved file.')

