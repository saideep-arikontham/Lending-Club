#!/usr/bin/env python
# coding: utf-8

# In[30]:


#reading individual applicant data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data/individual_application_information.csv').set_index('id')
df


# In[31]:


#Changed 0+ years to 0.
#Changing it to 0.5 now.
df.loc[df['emp_length']==0, 'emp_length'] = 0.5
df['emp_length'].value_counts()


# In[32]:


print(f'''loan amnt ! = funded_amnt rows : {df[(df['loan_amnt'] != df['funded_amnt'])].shape[0]}''')
print(f'''loan amnt ! = funded_amnt_inv rows : {df[(df['loan_amnt'] != df['funded_amnt_inv'])].shape[0]}''')

#dropping funded_amnt
df.drop(['funded_amnt'], axis = 1, inplace = True)
print('\n- Dropped funded_amnt')

#dropping addr_state
df.drop(['addr_state'], axis = 1, inplace = True)
print('- Dropped addr_state')

# disbursement_method, dropping this column
df.drop(['disbursement_method'], axis=1, inplace=True)
print('- Dropped disbursement_method')

print(f'\nNumber of Rows: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')


# ### Feature categorization

# In[33]:


category_cols = []
numeric_cols = []
for i in df.columns:
    if((df[i].dtype !='float64') & (df[i].dtype !='int64')):
        category_cols.append(i)
    else:
        numeric_cols.append(i)
print(category_cols)


# In[34]:


target = 'loan_status'
category_cols.remove(target)


# In[35]:


for i in category_cols:
    print(i, len(df[i].unique()))
print(target, len(df[target].unique()))


# ### Encoding

# In[36]:


print(f'Total categories before dropping: {len(category_cols)+1}')
for i in category_cols + [target]:
    print(f'- {i}, {len(df[i].unique())}')


# In[37]:


# Examine the rest of the columns closely.

# i. Grade
# This is assigned by lending club as the risk factor. Therefore A has the least risk, B has Highest risk.
#Assign values 10,20,30,... which we can consider as risk score - high value meaning high risk

grade_mapping = {'A': 0, 'B': 10, 'C': 20, 'D': 30, 'E': 40, 'F': 50, 'G': 60}
# Apply the mapping to the 'color' column
df['grade'] = df['grade'].map(grade_mapping)
print(f'Mapping used for grade:\n {grade_mapping}\n')

# ii. Subgrade
# Just add the grade encoded value above with the subgrade category i.e., if we have A5 then 10+5
df['sub_grade'] = df['sub_grade'].str.get(1).astype(int) + df['grade']
print('Sub grade is calculated based on grade encoding.\n')

# iii. Home ownership. The following are manually mapped encoding values that will be used
own_mapping = {'NONE':0, 'OTHER':1, 'ANY':2, 'MORTGAGE':3, 'RENT':4, 'OWN':5}
df['home_ownership'] = df['home_ownership'].map(own_mapping)
print(f'Mapping used for home ownership:\n {own_mapping}\n')

# iv. verification_status. The following are manually mapped encoding values that will be used
verification_mapping = {'Not Verified':0, 'Verified':1, 'Source Verified':2}
df['verification_status'] = df['verification_status'].map(verification_mapping)
print(f'Mapping used for verification_status:\n {verification_mapping}\n')


# In[38]:


from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import LabelBinarizer

# v. initial_list_status, using label binarizer
label_binarizer = LabelBinarizer()
df['initial_list_status'] = label_binarizer.fit_transform(df['initial_list_status'])

# Create a mapping dictionary
mapping_dict = dict(zip(label_binarizer.classes_, range(len(label_binarizer.classes_))))

print("Mapping used for initial_list_status:")
print(mapping_dict)


# In[39]:


# vi. Purpose. Using target encoder
encoder = TargetEncoder()
df['purpose'] = encoder.fit_transform(df[['purpose']], df['loan_status'])
print('Encoding for feature "purpose" - done')


# In[40]:


#Target

# vii. loan_status, paid = 1, charged off = 0
target_mapping = {'Charged Off':0, 'Fully Paid':1}
df['loan_status'] = df['loan_status'].map(target_mapping)
print(f'Mapping used for loan_status:\n {target_mapping}\n')


# In[41]:


for i in df.columns:
    if(df[i].dtype != 'float64' and df[i].dtype != 'int64'):
        print('Encoding not done properly. Recheck')
        break
else:
    print('Encoding done')


# In[42]:


df.shape[1] == len(numeric_cols) + len(category_cols) + 1


# ### Outlier understanding - Manual removal

# In[45]:


dup = df.copy()


# In[46]:


#removing outliers from installment:
sns.set(style="darkgrid")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))  # 1 row, 2 columns

axes[0].scatter(round(dup['installment']), dup['funded_amnt_inv'],alpha=0.2)
axes[0].set_title('Installment - Before')
axes[0].set_xlabel('installment')
axes[0].set_ylabel('loan amount')



x = dup['int_rate'] / (12 * 100)


# EMI formula
dup['emi'] = dup['funded_amnt_inv'] * (x * (1 + x) ** dup['term_months']) / \
      ((1 + x) ** dup['term_months'] - 1)


dup = dup[abs(dup['emi'] - dup['installment']) < 30]
dup.drop(['emi'], axis=1, inplace=True)


axes[1].scatter(round(dup['installment']), dup['funded_amnt_inv'],alpha=0.2)
axes[1].set_title('Installment - After')
axes[1].set_xlabel('installment')
axes[1].set_ylabel('loan amount')

plt.savefig('figs/installment_before_after.png')
plt.show()


# In[47]:


#removing outliers from annual income
sns.set(style="darkgrid")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))  # 1 row, 2 columns

# Plot on the first subplot (left)
axes[0].scatter(dup['annual_inc'], dup['funded_amnt_inv'],alpha=0.2)
axes[0].set_title('Annual income - Before')
axes[0].set_xlabel('annual income')
axes[0].set_ylabel('loan amount')


dup = dup[dup['annual_inc'] <= 1500000]

# Plot on the second subplot (right)
axes[1].scatter(dup['annual_inc'], dup['funded_amnt_inv'],alpha=0.2)
axes[1].set_title('Annual income - After')
axes[1].set_xlabel('annual income')
axes[1].set_ylabel('loan amount')

# Adjust layout for better spacing
plt.savefig('figs/annual_income_before_after.png')
plt.show()


# In[17]:


#removing outliers from delinq_2yrs
print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['delinq_2yrs'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('deliquent incidents')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['delinq_2yrs'] <= 15]


print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['delinq_2yrs'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('deliquent incidents')
plt.ylabel('loan amount')
plt.show()


# In[18]:


#removing outliers from inq_last_6mths
print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['inq_last_6mths'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('No. of inquiries')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['inq_last_6mths']<=6]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['inq_last_6mths'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('No. of inquiries')
plt.ylabel('loan amount')
plt.show()


# In[19]:


#removing outliers from open_acc
print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['open_acc'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('No. of open credit lines')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['open_acc']<=50]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['open_acc'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('No. of open credit lines')
plt.ylabel('loan amount')
plt.show()


# In[20]:


#removing outliers from pub_rec

print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['pub_rec'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('No. of public records')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['pub_rec']<=15]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['pub_rec'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('No. of public records')
plt.ylabel('loan amount')
plt.show()


# In[21]:


#removing outliers from total_acc

print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['total_acc'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('No. of credit lines')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['total_acc'] <= 100]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['total_acc'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('No. of credit lines')
plt.ylabel('loan amount')
plt.show()


# In[22]:


#removing outliers from collections_12_mths_ex_med

print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['collections_12_mths_ex_med'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('No. of collections excluding medication')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['collections_12_mths_ex_med'] <= 5]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['collections_12_mths_ex_med'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('No. of collections excluding medication')
plt.ylabel('loan amount')
plt.show()


# In[23]:


#removing outliers from tot_coll_amt

print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['tot_coll_amt'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('total collection amount')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['tot_coll_amt'] < 400000]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['tot_coll_amt'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('total collection amount')
plt.ylabel('loan amount')
plt.show()


# In[24]:


#removing outliers from tot_cur_bal

print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['tot_cur_bal'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('total current balance')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['tot_cur_bal'] < 3000000]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['tot_cur_bal'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('total current balance')
plt.ylabel('loan amount')
plt.show()


# In[25]:


#removing outliers from tot_cur_bal

print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['avg_cur_bal'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('avg current balance')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['avg_cur_bal'] < 300000]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['avg_cur_bal'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('avg current balance')
plt.ylabel('loan amount')
plt.show()


# In[26]:


print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['bc_util'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('bc_util')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['bc_util'] < 150]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['bc_util'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('bc_util')
plt.ylabel('loan amount')
plt.show()


# In[27]:


print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['chargeoff_within_12_mths'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('chargeoff_within_12_mths')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['chargeoff_within_12_mths'] <5]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['chargeoff_within_12_mths'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('chargeoff_within_12_mths')
plt.ylabel('loan amount')
plt.show()


# In[28]:


print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['delinq_amnt'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('delinq_amnt')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['delinq_amnt'] <= 60000]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['delinq_amnt'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('delinq_amnt')
plt.ylabel('loan amount')
plt.show()


# In[29]:


print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['num_accts_ever_120_pd'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('num_accts_ever_120_pd')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['num_accts_ever_120_pd'] <= 30]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['num_accts_ever_120_pd'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('num_accts_ever_120_pd')
plt.ylabel('loan amount')
plt.show()


# In[30]:


print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['num_tl_op_past_12m'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('num_tl_op_past_12m')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['num_tl_op_past_12m'] <= 20]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['num_tl_op_past_12m'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('num_tl_op_past_12m')
plt.ylabel('loan amount')
plt.show()


# In[31]:


print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['pct_tl_nvr_dlq'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('pct_tl_nvr_dlq')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['pct_tl_nvr_dlq'] > 20]

print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['pct_tl_nvr_dlq'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('pct_tl_nvr_dlq')
plt.ylabel('loan amount')
plt.show()


# In[32]:


print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['tot_hi_cred_lim'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('tot_hi_cred_lim')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['tot_hi_cred_lim'] < 4000000]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['tot_hi_cred_lim'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('tot_hi_cred_lim')
plt.ylabel('loan amount')
plt.show()


# In[33]:


print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['total_bal_ex_mort'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('total_bal_ex_mort')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['total_bal_ex_mort'] < 1000000]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['total_bal_ex_mort'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('total_bal_ex_mort')
plt.ylabel('loan amount')
plt.show()


# In[34]:


print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['total_bc_limit'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('total_bc_limit')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['total_bc_limit'] < 300000]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['total_bc_limit'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('total_bc_limit')
plt.ylabel('loan amount')
plt.show()


# In[35]:


print('Before:')
sns.set(style="darkgrid")
plt.scatter(dup['total_il_high_credit_limit'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('total_il_high_credit_limit')
plt.ylabel('loan amount')
plt.show()

dup = dup[dup['total_il_high_credit_limit'] < 800000]

print('After:')
sns.set(style="darkgrid")
plt.scatter(dup['total_il_high_credit_limit'], dup['funded_amnt_inv'],alpha=0.2)
plt.xlabel('total_il_high_credit_limit')
plt.ylabel('loan amount')
plt.show()


# In[36]:


df.shape[0] - dup.shape[0]


# In[37]:


df = dup.copy()


# ### Imputing

# In[38]:


df[category_cols+['loan_status']].isna().sum()


# In[39]:


# impute numeric cols

columns_to_impute = []
for i in numeric_cols:
    if(df[i].isna().sum()>0):
        columns_to_impute.append(i)
        print(i, df[i].isna().sum())



# In[40]:


import missingno

missingno.matrix(df[columns_to_impute])


# In[41]:


from sklearn.impute import KNNImputer

rest = list(set(df.columns.tolist()).difference(columns_to_impute))

for i in columns_to_impute:
    # Instantiate the KNNImputer
    knn_imputer = KNNImputer(n_neighbors=3)

    # Impute missing values in the entire DataFrame
    df[rest+[i]] = knn_imputer.fit_transform(df[rest+[i]])
    print(f'- Imputed {i}')

df


# In[42]:


df[columns_to_impute].describe()


# In[43]:


df.to_csv('data/imputed_data.csv')

