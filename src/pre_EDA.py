#!/usr/bin/env python
# coding: utf-8

# In[1]:


#reading individual applicant data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data/individual_application_information.csv').set_index('id')
df


# ### About Loan Status

# In[2]:


count = df['loan_status'].value_counts()
display(count)

sns.set(style="darkgrid")
plt.pie(count, labels = count.index, autopct='%1.2f%%', startangle=0)
plt.title('Loan status - breakdown')
plt.show()
plt.savefig('figs/loan_status_breakdown.png')


# ### grade vs interest rate

# In[3]:


p = df[['grade','int_rate']].groupby('grade').count().rename({'int_rate':'count'}, axis=1)
display(p)
sns.set(style="darkgrid")
plt.pie(p['count'], labels = p.index, autopct='%1.1f%%', startangle=0)
plt.title('Grade - breakdown')
plt.savefig('figs/grade_breakdown.png')
plt.show()


del p

sns.set(style="darkgrid")
plt.plot(df[['grade','int_rate']].groupby('grade').mean()['int_rate'], marker = 'o', label = 'avg interest rate')
plt.plot(df[['grade','int_rate']].groupby('grade').min()['int_rate'], marker = 'x', label = 'min interest rate')
plt.plot(df[['grade','int_rate']].groupby('grade').max()['int_rate'], marker = '^', label = 'max interest rate')
plt.legend()
plt.xlabel('Grade')
plt.ylabel('Interest rate')
plt.title('Interest rate vs Grade')
plt.savefig('figs/int_vs_grade.png')
plt.show()


grade_counts = df.groupby(['grade', 'loan_status']).size().unstack(fill_value=0)
display(grade_counts)

# Plot the results
sns.set(style="darkgrid")
grade_counts.plot(kind='bar', color=['#ff7f0e','#1f77b4'])
plt.title('loan_status per grade')
plt.xlabel('Grade')
plt.ylabel('loan status count (log scale)')
plt.legend(title='loan_status')
plt.yscale('log')
plt.savefig('figs/loan_status_per_grade.png')

plt.show()

#calculating percentages
grade_counts['Charged_off_percentage'] = grade_counts['Charged Off'] / (grade_counts['Charged Off']+grade_counts['Fully Paid'])* 100
grade_counts['fully_paid_percentage'] = grade_counts['Fully Paid'] / (grade_counts['Charged Off']+grade_counts['Fully Paid'])* 100

display(grade_counts[['Charged_off_percentage','fully_paid_percentage']])

del grade_counts


# ### purpose of loan

# In[4]:


purpose = df[['purpose','loan_status']].groupby('purpose').count().rename({'loan_status':'count'}, axis=1)
purpose['percentage'] = purpose['count'] / purpose['count'].sum() * 100
purpose

p = df.groupby(['purpose', 'loan_status']).size().unstack(fill_value=0)
display(p)

# Plot the results
sns.set(style="darkgrid")
p.plot(kind='bar', color=['#ff7f0e','#1f77b4'])
plt.title('loan_status per purpose')
plt.xlabel('Purpose')
plt.ylabel('loan status count (log scale)')
plt.legend(title='loan_status')
plt.yscale('log')
plt.savefig('figs/loan_status_per_purpose.png')
plt.show()

#calculating percentages
p['Charged_off_percentage'] = p['Charged Off'] / (p['Charged Off']+p['Fully Paid'])* 100
p['fully_paid_percentage'] = p['Fully Paid'] / (p['Charged Off']+p['Fully Paid'])* 100

display(p[['Charged_off_percentage','fully_paid_percentage']])

del p


# ### Examining loan amount relation to loan_status

# In[5]:


target = 'loan_status'

loan_amnt_sum = df[['loan_amnt']].describe().astype('str')
display(loan_amnt_sum)

#creating a breakdown of loan amount, whether its less than mean or greater than mean
mean_amnt = round(float(loan_amnt_sum.loc['mean','loan_amnt']))

loan_amnt_breakdown = df[['loan_amnt',target]]

conditions = [
    (loan_amnt_breakdown['loan_amnt'] < mean_amnt),
    (loan_amnt_breakdown['loan_amnt'] >= mean_amnt)
     ]

#classes of breakdown
values = [f'less than {mean_amnt}$',
          f'more than {mean_amnt}$']

# Apply conditions using numpy.select
loan_amnt_breakdown['breakdown'] = np.select(conditions, values, default=np.nan)

display(loan_amnt_breakdown)

g1 = loan_amnt_breakdown.groupby(['breakdown', target]).size().unstack(fill_value=0)

display(g1)

# Plot the results
sns.set(style="darkgrid")
g1.plot(kind='bar', color=['#ff7f0e','#1f77b4'])
plt.title('breakdown of loan_status w.r.t. loan amount')
plt.xlabel('Loan amount')
plt.ylabel('Count')
plt.legend(title='Loan status')
plt.savefig('figs/loan_status_and_loan_amnt.png')
plt.show()

#calculating percentages
print('Loan charged off percentage when loan amount:')
g1['charged_off_percentage'] = g1['Charged Off'] / (g1['Charged Off'] + g1['Fully Paid'])* 100
g1['fully_paid_percentage'] = g1['Fully Paid'] / (g1['Charged Off'] + g1['Fully Paid'])* 100

display(g1[['fully_paid_percentage','charged_off_percentage']])

del g1
del loan_amnt_breakdown
del loan_amnt_sum


# ### Examining interest rates

# In[6]:


int_sum = df[['int_rate']].describe().astype('str')
display(int_sum)


#creating a breakdown of interest rate, whether its less than mean or greater than mean
mean_int = round(float(int_sum.loc['mean','int_rate']))

int_breakdown = df[['int_rate',target]]

conditions = [
    (int_breakdown['int_rate'] < mean_int),
    (int_breakdown['int_rate'] >= mean_int)
     ]

#classes of breakdown
values = [f'less than {mean_int}',
          f'more than {mean_int}']

# Apply conditions using numpy.select
int_breakdown['breakdown'] = np.select(conditions, values, default=np.nan)

display(int_breakdown)

g2 = int_breakdown.groupby(['breakdown', target]).size().unstack(fill_value=0)

display(g2)

# Plot the results
sns.set(style="darkgrid")
g2.plot(kind='bar', color=['#ff7f0e','#1f77b4'])
plt.title('breakdown of loan_status w.r.t. loan interest')
plt.xlabel('Loan interest')
plt.ylabel('Count')
plt.legend(title='Loan status')
plt.savefig('figs/loan_status_and_interest.png')
plt.show()

#calculating percentages
print('Loan charged off percentage when loan interest rate:')
g2['charged_off_percentage'] = g2['Charged Off'] / (g2['Charged Off'] + g2['Fully Paid'])* 100
g2['fully_paid_percentage'] = g2['Fully Paid'] / (g2['Charged Off'] + g2['Fully Paid'])* 100

display(g2[['fully_paid_percentage','charged_off_percentage']])

del g2
del int_breakdown
del int_sum


# ### Examining employment length

# In[7]:


#calculating fully paid and charged off per employment length
emp_len_count = df.groupby(['emp_length', 'loan_status']).size().unstack(fill_value=0)
display(emp_len_count)

# Plot the results
sns.set(style="darkgrid")
emp_len_count.plot(kind='bar', color=['#ff7f0e','#1f77b4'])
plt.title('loan_status per employment duration')
plt.xlabel('Employment length (years)')
plt.ylabel('loan status count (log scale)')
plt.legend(title='loan_status')
plt.yscale('log')
plt.savefig('figs/loan_status_per_emp_length.png')
plt.show()

#calculating percentages
emp_len_count['Charged_off_percentage'] = emp_len_count['Charged Off'] / (emp_len_count['Charged Off']+emp_len_count['Fully Paid'])* 100
emp_len_count['fully_paid_percentage'] = emp_len_count['Fully Paid'] / (emp_len_count['Charged Off']+emp_len_count['Fully Paid'])* 100

display(emp_len_count[['Charged_off_percentage','fully_paid_percentage']])


del emp_len_count


# ### Examining number of mortgage accounts

# In[8]:


#creating a breakdown of mortgage account number, whether its less than 2 or greater than 2
threshold = 2

mort_breakdown = df[['mort_acc',target]]

conditions = [
    (mort_breakdown['mort_acc'] < threshold),
    (mort_breakdown['mort_acc'] >= threshold)
     ]

#classes of breakdown
values = [f'less than {threshold}',
          f'more than {threshold}']

# Apply conditions using numpy.select
mort_breakdown['breakdown'] = np.select(conditions, values, default=np.nan)

display(mort_breakdown)

g3 = mort_breakdown.groupby(['breakdown', target]).size().unstack(fill_value=0)

display(g3)

# Plot the results
sns.set(style="darkgrid")
g3.plot(kind='bar', color=['#ff7f0e','#1f77b4'])
plt.title('breakdown of loan_status w.r.t. mortgage account number')
plt.xlabel('Number of mortgage accounts')
plt.ylabel('Count')
plt.legend(title='Loan status')
plt.savefig('figs/loan_status_and_mortgage_acc.png')
plt.show()

#calculating percentages
print('Loan charged off percentage when loan interest rate:')
g3['charged_off_percentage'] = g3['Charged Off'] / (g3['Charged Off'] + g3['Fully Paid']) * 100
g3['fully_paid_percentage'] = g3['Fully Paid'] / (g3['Charged Off'] + g3['Fully Paid'])* 100

display(g3[['fully_paid_percentage','charged_off_percentage']])

del g3
del mort_breakdown


# In[9]:


inc_sum = df[['annual_inc']].describe().astype('str')
display(inc_sum)


#creating a breakdown of interest rate, whether its less than mean or greater than mean
mean_inc = round(float(inc_sum.loc['mean','annual_inc']))

inc_breakdown = df[['annual_inc',target]]

conditions = [
    (inc_breakdown['annual_inc'] < mean_inc),
    (inc_breakdown['annual_inc'] >= mean_inc)
     ]

#classes of breakdown
values = [f'less than {mean_inc}$',
          f'more than {mean_inc}$']

# Apply conditions using numpy.select
inc_breakdown['breakdown'] = np.select(conditions, values, default=np.nan)

display(inc_breakdown)

g4 = inc_breakdown.groupby(['breakdown', target]).size().unstack(fill_value=0)

display(g4)

# Plot the results
sns.set(style="darkgrid")
g4.plot(kind='bar', color=['#ff7f0e','#1f77b4'])
plt.title('breakdown of loan_status w.r.t. annual income')
plt.xlabel('annual income')
plt.ylabel('Count')
plt.legend(title='Loan status')
plt.savefig('figs/loan_status_and_income.png')
plt.show()

#calculating percentages
print('Loan charged off percentage when annual income:')
g4['charged_off_percentage'] = g4['Charged Off'] / (g4['Charged Off'] + g4['Fully Paid'])* 100
g4['fully_paid_percentage'] = g4['Fully Paid'] / (g4['Charged Off'] + g4['Fully Paid'])* 100

display(g4[['fully_paid_percentage','charged_off_percentage']])

del g4
del inc_breakdown
del inc_sum

