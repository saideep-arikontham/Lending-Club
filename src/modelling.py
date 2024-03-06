#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('data/removed_outlier.csv').set_index('id')
df


# In[2]:


df['loan_status'].value_counts() / df.shape[0] *100


# In[3]:


df['loan_status'].value_counts() 


# In[4]:


from sklearn.metrics import classification_report



def get_scores(model, X_train,y_train, X_test, y_test):
    # Make predictions on the training set
    sns.set(style="darkgrid")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))  # 1 row, 2 columns

    y_pred = model.predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)

    print("- Train Accuracy:", accuracy)
    print("- Train Precision:", precision)
    print("- Train Recall:", recall)
    print("- Train F1 Score:", f1)
    
    #print(classification_report(y, y_pred))

    # Calculate confusion matrix
    cm1 = confusion_matrix(y_train, y_pred)

    # Plot the confusion matrix
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax = axes[0])
    axes[0].set_title("Confusion Matrix - Train")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n- Test Accuracy:", accuracy)
    print("- Test Precision:", precision)
    print("- Test Recall:", recall)
    print("- Test F1 Score:", f1)
    
    #print(classification_report(y, y_pred))

    # Calculate confusion matrix
    cm2 = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", ax = axes[1])
    axes[1].set_title("Confusion Matrix - Test")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.show()


# # Imbalanced data

# In[5]:


# Separate features (X) and labels (y)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Assuming X, y are your features and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)


# ## Logistic Regression

# In[6]:


from sklearn.linear_model import LogisticRegression

# Create an XGBoost classifier for binary classification
clf = LogisticRegression(C = 100,random_state = 42)

#Model training
clf.fit(X_train, y_train)

#getting scores
get_scores(clf, X_train, y_train, X_test, y_test)


# ## DecisionTreeClassifier

# In[7]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 9, random_state = 42)

#Model training
clf.fit(X_train, y_train)

get_scores(clf, X_train, y_train, X_test, y_test)


# ## Naive Bayes Classifier

# In[8]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#Model training
clf.fit(X_train, y_train)

get_scores(clf, X_train, y_train, X_test, y_test)


# ## XGBoost

# In[9]:


from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=1000, max_depth = 9, random_state = 42, objective='binary:logistic', learning_rate = 0.5)

#Model training
clf.fit(X_train, y_train)

get_scores(clf, X_train, y_train, X_test, y_test)


# # Undersampling

# In[10]:


from imblearn.under_sampling import RandomUnderSampler

# Separate features (X) and labels (y)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"- Class distribution before undersampling:\n{pd.Series(y_train).value_counts()}")

# Apply NearMiss to the training set
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

print(f"- Class distribution after undersampling:\n{pd.Series(y_train_resampled).value_counts()}")


# ## Logistic Regression

# In[11]:


# Create an XGBoost classifier for binary classification
clf = LogisticRegression(C = 100,random_state = 42)

#Model training
clf.fit(X_train_resampled, y_train_resampled)

get_scores(clf, X_train_resampled, y_train_resampled, X_test, y_test)


# ## DecisionTreeClassifier

# In[12]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 9, random_state = 42)

#Model training
clf.fit(X_train_resampled, y_train_resampled)

get_scores(clf, X_train_resampled, y_train_resampled, X_test, y_test)


# ## Naive Bayes

# In[13]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#Model training
clf.fit(X_train_resampled, y_train_resampled)

get_scores(clf, X_train_resampled, y_train_resampled, X_test, y_test)


# ## XGBoost

# In[14]:


from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=1000, max_depth = 9, random_state = 42, objective='binary:logistic', learning_rate = 0.5)

#Model training
clf.fit(X_train_resampled, y_train_resampled)

get_scores(clf, X_train_resampled, y_train_resampled, X_test, y_test)


# # Over Sampling

# In[15]:


from imblearn.over_sampling import SMOTE


# Separate features (X) and labels (y)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the class distribution before oversampling
print("Class distribution before oversampling:\n", pd.Series(y_train).value_counts())

# Apply SMOTE to the training set
oversampler = SMOTE(random_state=42, k_neighbors = 125, n_jobs = -1)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Print the class distribution after oversampling
print("Class distribution after oversampling:\n", pd.Series(y_train_resampled).value_counts())


# ## Logistic Regression

# In[16]:


clf = LogisticRegression(C = 100,random_state = 42)

#Model training
clf.fit(X_train_resampled, y_train_resampled)

get_scores(clf, X_train_resampled, y_train_resampled, X_test, y_test)


# ## DecisionTreeClassifier

# In[17]:


clf = DecisionTreeClassifier(max_depth = 9, random_state = 42)

#Model training
clf.fit(X_train_resampled, y_train_resampled)

get_scores(clf, X_train_resampled, y_train_resampled, X_test, y_test)


# ## Naive Bayes

# In[18]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#Model training
clf.fit(X_train_resampled, y_train_resampled)

get_scores(clf, X_train_resampled, y_train_resampled, X_test, y_test)


# ## XGBoost

# In[19]:


from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=1000, max_depth = 9, random_state = 42, objective='binary:logistic', learning_rate = 0.5)

#Model training
clf.fit(X_train_resampled, y_train_resampled)

get_scores(clf, X_train_resampled, y_train_resampled, X_test, y_test)

