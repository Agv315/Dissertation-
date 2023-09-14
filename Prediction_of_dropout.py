#!/usr/bin/env python
# coding: utf-8

# ### Part 1 Exploratory Data Analysis

# In[2]:


#Step 1:  load the dataset into a pandas dataframe
# Importing the libraries NUMPY, PANDAS, MATPLOT SEABOARN, Sckitlearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import warnings

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import log_loss


# In[3]:


myproject=pd.read_csv("D:\Research skills\Project_data.csv")
myproject


# In[4]:


# shape of the dataframe
print(myproject.shape)


# In[5]:


# Get the data types of each column
data_types = myproject.dtypes
data_types


# In[6]:


# checking for missing values 
missing_values = pd.isna(myproject).all(axis=0)
# Print the resulting boolean Series
missing_count = missing_values.sum()
print(missing_values)
print("Number of missing values:", missing_count)


# In[7]:


myproject=pd.read_csv("D:\Research skills\Project_data.csv")
myproject=myproject.dropna()
myproject


# In[8]:


#ETL Extract Transform Load


# In[9]:


#Transformation  of data fathers and mothers qualificaiton
# 2 - Higher Education - bachelor’s degree  3 - Higher Education – Degree
# 2 & 3 appear to be the same, hence 3 is replaced with 2
# the inconsistency was rectified
myproject = pd.DataFrame(myproject)

# Replace values
myproject['Father\'s qualification'] = myproject['Father\'s qualification'].replace(3, 2)
myproject['Mother\'s qualification'] = myproject['Mother\'s qualification'].replace(3, 2)
print(myproject[['Father\'s qualification']].head())
print(myproject[['Mother\'s qualification']].head())


# In[10]:


# Filter the rows based on your criteria
#  Dropping curricular unit 1st sem + curricular units 2nd sem = Graduate
myproject = myproject[~((myproject['Curricular units 1st sem (grade)'] == 0) &
                        (myproject['Curricular units 2nd sem (grade)'] == 0) &
                        (myproject['Target'] == 'Graduate'))]


# Reset the index 
myproject.reset_index(drop=True, inplace=True)
myproject


# In[ ]:





# In[11]:


#Distribution of the target variable
from matplotlib import style
targetvariable=myproject['Target'].value_counts().rename_axis('Status').reset_index(name='counts')
targetvariable


# In[12]:


Keys = ['Graduate (0)', 'Dropout (1)','Enrolled(2)']
values=targetvariable.counts
plt.pie(values, labels=Keys,startangle=180,autopct='%1.1f%%',colors = ['lightgreen', 'skyblue','grey'])
plt.axis('equal')
plt.legend(title='Legend',loc='lower right',bbox_to_anchor=(0,0,0,0))
plt.title('Percentage of Graduate vs Dropout vs Enrolled')
plt.show()


# In[13]:


#Binary classification of the target
myproject['New_Target'] = myproject['Target'].map({'Graduate': 'Non-Dropout', 'Enrolled': 'Non-Dropout', 'Dropout': 'Dropout'})
myproject.drop(columns=['Target'], inplace=True)
print(myproject[['New_Target']].head())


# In[14]:


newtargetvariable=myproject['New_Target'].value_counts().rename_axis('Status').reset_index(name='counts')
newtargetvariable


# In[15]:


new_targetvalues = myproject['New_Target'].unique()
new_targetcounts = myproject['New_Target'].value_counts()
print(new_targetvalues)
print(new_targetcounts)


# In[16]:


# New Target variable distribution
#Plotting the pie chart
plt.figure(figsize=(6, 6))
plt.pie(new_targetcounts, labels=new_targetcounts.index, autopct='%1.1f%%', startangle=180,colors = ['lightgreen', 'skyblue'])
plt.title('Percentage distribution of Dropout Status')
plt.axis('equal')
plt.legend(title='Dropout Status', labels=new_targetcounts.index, loc='lower right',bbox_to_anchor=(0,0,0,0))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


# Taking the count for gender 
gender_counts = myproject['Gender'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors = ['lightgreen', 'skyblue'], startangle=90)
#adding legend
plt.legend(
    title='Gender',
    labels=['Female (0)', 'Male (1)'], 
    loc='upper right',
    bbox_to_anchor=(1.2, 1)
)

plt.title('Gender Distribution at Polytechnic University,Portugal')

# Show the chart
plt.show()


# In[ ]:





# In[ ]:





# In[18]:


#  A new column Total Grade Average is added it  combines the grades of semester 1 and 2 
#Calculate the total grade average
myproject['Total Grade Average'] = (myproject['Curricular units 1st sem (grade)'] +
                                    myproject['Curricular units 2nd sem (grade)']) / 2
print(myproject[['Total Grade Average']].head())


# In[19]:


# Distribution of Total Grade Average 


# In[20]:


print(myproject['Total Grade Average'].describe())


# In[21]:


#Distribution of Total Grade Average 
sns.set_style("whitegrid")
plt.figure(figsize=(10,4))
sns.boxplot(x='Total Grade Average', data=myproject ,color='green')
plt.title('Distribution of Total Grade Average')
plt.xlabel('Grade')
plt.show()


# In[22]:


#grade_stats = myproject['Total Grade Average'].describe()
#grade_stats


# In[23]:


# Display basic statistics of age
print(myproject['Age at enrollment'].describe())


# In[24]:


#Distribution of Age at Enrolment 
plt.figure(figsize=(10,4))
sns.boxplot(x='Age at enrollment', data=myproject ,color='green')
plt.title('Distribution of Age Enrolment')
plt.xlabel('Age')
plt.show()


# In[25]:


# Define the bins for age intervals
bins = [17, 25, 35, 45, 55, 65, 70,75]

# Define labels for each interval
labels = ['17-25', '26-35', '36-45', '46-55', '56-65', '66-70','71-75']

# Use pd.cut() to create the new column "Age Interval"
myproject['Age Interval'] = pd.cut(myproject['Age at enrollment'], bins=bins, labels=labels, right=False)
#print(myproject.head(5))
print(myproject[['Age at enrollment', 'Age Interval']].head())


# In[26]:


age_interval_counts = myproject['Age Interval'].value_counts()
age_interval_counts


# In[27]:


age_interval_counts = myproject['Age Interval'].value_counts()

# Calculate the total count
total_count = age_interval_counts.sum()

# Calculate the percentage distribution
age_interval_percentages = (age_interval_counts / total_count) * 100

#  a bar chart showing both counts and percentages
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=age_interval_counts.index, y=age_interval_counts, palette='Greens_r', alpha=0.7, label='Count')
ax2 = ax.twinx()  # a secondary y-axis for percentages
ax2.plot(age_interval_counts.index, age_interval_percentages, color='red', marker='o', label='Percentage')

plt.title('Percentag-Age Interval Distribution')
ax.set_xlabel('Age Interval')
ax.set_ylabel('Count')
ax2.set_ylabel('Percentage (%)')
ax.set_xticklabels(age_interval_counts.index, rotation=45) 
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))


plt.show()


# In[28]:


#Descriptive Analysis of the  categorical variables 


# In[29]:


List2_categoricalvariables = ['Debtor','Tuition fees up to date','Scholarship holder','International']
plt.figure(figsize=(12, 8))

value_labels = {
    'Debtor': {1: 'Yes', 0: 'No'},
    'Tuition fees up to date': {1: 'Yes', 0: 'No'},
    'Scholarship holder': {1: 'Yes', 0: 'No'},
    'International':{1: 'Yes', 0: 'No'}
}

# a subplot grid
plt.figure(figsize=(10, 6))

# Loop through the categorical variables
for i, col in enumerate(List2_categoricalvariables, 1):
    plt.subplot(2, 2, i)
    plt.title(col)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # the value_labels mapping to label the x-axis
    plot = sns.countplot(data=myproject, x=col, hue="New_Target", palette=['green', 'skyblue'], linewidth=1)
    plt.xticks(ticks=[0, 1], labels=[value_labels[col][0], value_labels[col][1]])
    plt.xlabel(col)
    plot.legend(loc='upper left', bbox_to_anchor=(1, 1))


plt.tight_layout()
plt.show()


# In[30]:


# List of categorical variables
List1_categoricalvariables = ['Displaced', 'Educational special needs', 'Gender']

# Mapping of values to labels
value_labels = {
    'Displaced': {1: 'Yes', 0: 'No'},
    'Educational special needs': {1: 'Yes', 0: 'No'},
    'Gender': {1: 'Male', 0: 'Female'}
}

#  a subplot grid
plt.figure(figsize=(10, 6))

# Loop through the categorical variables
for i, col in enumerate(List1_categoricalvariables, 1):
    plt.subplot(2, 2, i)
    plt.title(col)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # Use the value_labels mapping to label the x-axis
    plot = sns.countplot(data=myproject, x=col, hue="New_Target", palette=['green', 'skyblue'], linewidth=1)
    plt.xticks(ticks=[0, 1], labels=[value_labels[col][0], value_labels[col][1]])
    plt.xlabel(col)
    plot.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 
plt.tight_layout()
plt.show()


# In[31]:


# Define the age intervals for the x-axis
age_intervals = ['17-25', '26-35', '36-45', '46-55', '56-65', '66-70','71-75']

# Filter 'Age Interval' values not in the specified intervals (if any)
filtered_data = myproject[myproject['Age Interval'].isin(age_intervals)]

# Create a cross-tabulation of 'Age Interval' against 'New_target'
cross_tab = pd.crosstab(filtered_data['Age Interval'], filtered_data['New_Target'])

# Plot the grouped bar plot
cross_tab.plot(kind='bar', color=['lightgreen', 'skyblue'], figsize=(10, 6))
plt.xlabel('Age Interval')
plt.ylabel('Count')
plt.title('New_target vs Age Interval')
plt.legend(title='New_target', loc='upper right', labels=['Non Drop Out', 'Drop Out'])

# Set custom x-axis tick labels
plt.xticks(range(len(age_intervals)), age_intervals, rotation=0)

plt.show()


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns

# Selecting numeric features
numeric_features = myproject.select_dtypes(include=['int64', 'float64'])
num_cols = len(numeric_features.columns)
num_rows = (num_cols + 2) // 3  

plt.figure(figsize=(15, 5 * num_rows))  

for i, column in enumerate(numeric_features.columns):
    plt.subplot(num_rows, 3, i + 1)
    plt.hist(myproject[column], bins=10, color='green')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()


# In[33]:


print(myproject['Previous qualification (grade)'].describe())


# In[34]:



# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the histogram with a distribution curve
sns.histplot(myproject['Age at enrollment'], kde=True, color='green')

# Add labels and a title
plt.xlabel('Age at Enrollment')
plt.ylabel('Frequency')
plt.title('Distribution of Age at Enrollment')

# Show the plot
plt.show()


# In[35]:




# Filter the DataFrame
filtered_df = myproject[myproject['New_Target'] == 'Non-Dropout'][['Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']]

# Set the figure size
plt.figure(figsize=(12, 6))

# Create the first subplot with histogram and distribution curve for 1st Semester Grades
plt.subplot(1, 2, 1)
sns.histplot(filtered_df['Curricular units 1st sem (grade)'], bins=10, kde=True, color='skyblue')
plt.title('Histogram of 1st Semester Grades')
plt.xlabel('Grades')
plt.ylabel('Frequency')

# Create the second subplot with histogram and distribution curve for 2nd Semester Grades
plt.subplot(1, 2, 2)
sns.histplot(filtered_df['Curricular units 2nd sem (grade)'], bins=10, kde=True, color='green')
plt.title('Histogram of 2nd Semester Grades')
plt.xlabel('Grades')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


myproject.skew(axis = 0, skipna = True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Part 2: Application of MLA

# ## Classification I

# In[55]:


# Splitting data into features (X) and target (y)
X = myproject.drop(['New_Target','Age Interval'], axis=1)
y = myproject['New_Target']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Lists to store metrics
model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
# Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'Gradient Boost': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Neural Network': MLPClassifier(),
}

# Iterate over models
for model_name, model in models.items():
    print(f"Training and evaluating {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Dropout',zero_division=1)
    recall = recall_score(y_test, y_pred, pos_label='Dropout')
    f1 = f1_score(y_test, y_pred, pos_label='Dropout')
    
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(f"{model_name} Precision: {precision:.2f}")
    print(f"{model_name} Recall: {recall:.2f}")
    print(f"{model_name} F1-Score: {f1:.2f}")
    
    
   
    # Generate classification report
    class_report = classification_report(y_test, y_pred, target_names=['Non-Dropout', 'Dropout'])
    #class_report_imbalanced = classification_report_imbalanced(y_test, y_pred, zero_division=1)
    #print(f"Imbalanced Classification Report for {model_name}:\n{class_report_imbalanced}")
    print(f"Classification Report for {model_name}:\n{class_report}")
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', annot_kws={"size": 14})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    
    print("=" * 40)
    
    print("=" * 40)
    # Append metrics to lists
    model_names.append(model_name)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    # Set the positions and width for the bars
positions = np.arange(len(model_names))
width = 0.2

# Create a grouped bar plot
plt.figure(figsize=(10, 6))
plt.bar(positions - width, accuracies, width, label='Accuracy', color='blue')
plt.bar(positions, precisions, width, label='Precision', color='green')
plt.bar(positions + width, recalls, width, label='Recall', color='orange')
plt.bar(positions + 2 * width, f1_scores, width, label='F1-Score', color='red')

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.xticks(positions, model_names, rotation=45)
plt.tight_layout()

plt.show()


# In[ ]:





# In[ ]:





# In[38]:


#Training Loss plots for Classification I  Neural Network


# In[44]:


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'Gradient Boost': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Neural Network': MLPClassifier(max_iter=1000),  # Increased max_iter for better convergence
}

# Iterate over models
for model_name, model in models.items():
    print(f"Training and evaluating {model_name}...")
    if model_name == 'Neural Network':
        # For neural networks, tracking  training loss and validation loss
        train_losses = []
        val_losses = []
        epochs = []
        for epoch in range(1, 51):  # Adjusted the number of epochs as rqd
            model.partial_fit(X_train, y_train, classes=np.unique(y))
            y_pred = model.predict(X_test)
            train_loss = model.loss_
            train_losses.append(train_loss)
            val_loss = log_loss(y_test, model.predict_proba(X_test))
            val_losses.append(val_loss)
            epochs.append(epoch)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
        
        # Plot the training loss graph , Validation loss
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - {model_name}')
        plt.legend()
        plt.show()
        
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


# In[45]:


# Plotting ROC for the other models 


# In[46]:


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define models
models = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'Gradient Boost': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Neural Network': MLPClassifier(max_iter=1000)
}

# Plot ROC curves
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    model.fit(X_train, y_train_encoded)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_prob)
    auc_score = roc_auc_score(y_test_encoded, y_pred_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()


# ### Classification II

# In[47]:


# BALANCING  THE DATASET USING SMOTE 


# In[48]:


# Variables are  assigned  SMOTE 
smote = SMOTE()
# Fittin and transformng the SMOTE object
X_smote, y_smote = smote.fit_resample(X_train,y_train)
# observing the target variable count and after SMOTE is performed
print(f'Myproject before SMOTE: {Counter(y)}')
print(f'Myproject after SMOTE: {Counter(y_smote)}')


# In[49]:


#Class distribution after SMOTE


# In[50]:


# Class distribution after SMOTE
class_distribution = {'Dropout': 2434, 'Non-Dropout': 2434}

# Specify colors
colors = ['skyblue', 'lightgreen']

# Create a pie chart
labels = class_distribution.keys()
sizes = class_distribution.values()

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Class Distribution After SMOTE')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()


# In[51]:


# Usingg the balanced dataset to train the data 


# In[52]:


X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)


# In[53]:


# the models are called using the previous function


# In[56]:


# Iterate over models
# Lists to store metrics
model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
for model_name, model in models.items():
    print(f"Training and evaluating {model_name}...")
    model.fit(X_train_smote, y_train_smote)
    y_pred_smote = model.predict(X_test_smote)
    
    accuracy = accuracy_score(y_test_smote, y_pred_smote)
    precision = precision_score(y_test_smote, y_pred_smote, pos_label='Dropout')
    recall = recall_score(y_test_smote, y_pred_smote, pos_label='Dropout')
    f1 = f1_score(y_test_smote, y_pred_smote, pos_label='Dropout')
    
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(f"{model_name} Precision: {precision:.2f}")
    print(f"{model_name} Recall: {recall:.2f}")
    print(f"{model_name} F1-Score: {f1:.2f}")
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, target_names=['Non-Dropout', 'Dropout'])
    #class_report_imbalanced = classification_report_imbalanced(y_test_smote, y_pred_smote, zero_division=1)
    #print(f"Imbalanced Classification Report for {model_name}:\n{class_report_imbalanced}")
    print(f"Classification Report for {model_name}:\n{class_report}")
    cm = confusion_matrix(y_test_smote, y_pred_smote)
    
    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', annot_kws={"size": 14})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    
    print("=" * 40)
    
    print("=" * 40)
    # Append metrics to lists
    model_names.append(model_name)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    # Set the positions and width for the bars
positions = np.arange(len(model_names))
width = 0.2

# Create a grouped bar plot
plt.figure(figsize=(10, 6))
plt.bar(positions - width, accuracies, width, label='Accuracy', color='blue')
plt.bar(positions, precisions, width, label='Precision', color='green')
plt.bar(positions + width, recalls, width, label='Recall', color='orange')
plt.bar(positions + 2 * width, f1_scores, width, label='F1-Score', color='red')

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.xticks(positions, model_names, rotation=45)
plt.tight_layout()

plt.show()


# In[ ]:


# Plotting the Training -Validation Loss after SMOTE for Neural Network 


# In[62]:


# Split data and encode labels
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_smote)
y_test_encoded = label_encoder.transform(y_test_smote)

# Define models
models = {
    'Neural Network': MLPClassifier(max_iter=1000),
}

# Plot training and validation loss for Neural Network
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    if model_name == 'Neural Network':
        train_losses = []
        val_losses = []
        epochs = []
        for epoch in range(1, 51):  # Adjusted the number of epochs as needed
            model.partial_fit(X_train_smote, y_train_encoded, classes=np.unique(y_train_encoded))
            y_pred_prob = model.predict_proba(X_test_smote)[:, 1]
            train_loss = model.loss_
            train_losses.append(train_loss)
            val_loss = log_loss(y_test_encoded, y_pred_prob)
            val_losses.append(val_loss)
            epochs.append(epoch)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

        plt.plot(epochs, train_losses, label=f'{model_name} Training Loss')
        plt.plot(epochs, val_losses, label=f'{model_name} Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()


# In[ ]:


# Plotting the ROC after SMOTE 


# In[63]:


# Splitting data into training and testing sets
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_smote)
y_test_encoded = label_encoder.transform(y_test_smote)


# Define models
models = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'Gradient Boost': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Neural Network': MLPClassifier(max_iter=1000)
}

# Plot ROC curves
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    model.fit(X_train_smote, y_train_encoded)
    y_pred_prob = model.predict_proba(X_test_smote)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_prob)
    auc_score = roc_auc_score(y_test_encoded, y_pred_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()


# ### Classification III

# In[ ]:


# Feature Selection 


# In[61]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
# the  model is fitted
model.fit(X_train_smote, y_train_smote)
# Getting the feature importances
importances=model.feature_importances_
# Sorting features according to importance
sort = importances.argsort()[::-1]
plt.figure(figsize=(12, 8))
# Plot the feature importances
plt.barh(X_train_smote.columns[sort],importances[sort],color='g')
plt.title("Feature importance using Random Forest" , fontsize = 8)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# In[ ]:


# Training the models  with feature selection using the most  important features


# In[ ]:


X_smote =X_smote[['Curricular units 2nd sem (approved)','Curricular units 1st sem (approved)',
                  'Tuition fees up to date','Curricular units 2nd sem (grade)',
                  'Total Grade Average','Curricular units 1st sem (grade)']]


# In[ ]:


X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)


# In[57]:


# Lists to store metrics
model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
for model_name, model in models.items():
    print(f"Training and evaluating {model_name}...")
    model.fit(X_train_smote, y_train_smote)
    y_pred_smote = model.predict(X_test_smote)
    
    accuracy = accuracy_score(y_test_smote, y_pred_smote)
    precision = precision_score(y_test_smote, y_pred_smote, pos_label='Dropout')
    recall = recall_score(y_test_smote, y_pred_smote, pos_label='Dropout')
    f1 = f1_score(y_test_smote, y_pred_smote, pos_label='Dropout')
    
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(f"{model_name} Precision: {precision:.2f}")
    print(f"{model_name} Recall: {recall:.2f}")
    print(f"{model_name} F1-Score: {f1:.2f}")
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, target_names=['Non-Dropout', 'Dropout'])
    #class_report_imbalanced = classification_report_imbalanced(y_test_smote, y_pred_smote, zero_division=1)
    #print(f"Imbalanced Classification Report for {model_name}:\n{class_report_imbalanced}")
    print(f"Classification Report for {model_name}:\n{class_report}")
    cm = confusion_matrix(y_test_smote, y_pred_smote)
    
    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', annot_kws={"size": 14})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    
    print("=" * 40)
    
    print("=" * 40)
    # Append metrics to lists
    model_names.append(model_name)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    # Set the positions and width for the bars
positions = np.arange(len(model_names))
width = 0.2

# Create a grouped bar plot
plt.figure(figsize=(10, 6))
plt.bar(positions - width, accuracies, width, label='Accuracy', color='blue')
plt.bar(positions, precisions, width, label='Precision', color='green')
plt.bar(positions + width, recalls, width, label='Recall', color='orange')
plt.bar(positions + 2 * width, f1_scores, width, label='F1-Score', color='red')

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.xticks(positions, model_names, rotation=45)
plt.tight_layout()

plt.show()


# In[ ]:


# Training Loss -Validation Plot after feature selection 


# In[59]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss  # You can use different loss functions for validation

# Define the selected features
selected_features = ['Curricular units 2nd sem (approved)', 'Curricular units 1st sem (approved)',
                     'Tuition fees up to date', 'Curricular units 2nd sem (grade)',
                     'Total Grade Average', 'Curricular units 1st sem (grade)']

# Select only the relevant columns from the training and test data
X_train_selected = X_train_smote[selected_features]
X_test_selected = X_test_smote[selected_features]

# Initialize the Neural Network model
model = MLPClassifier(max_iter=1, random_state=42, warm_start=True)

# Split a portion of the training data for validation
val_size = 0.2  # Adjust as needed
val_samples = int(X_train_selected.shape[0] * val_size)

X_train, X_val = X_train_selected[val_samples:], X_train_selected[:val_samples]
y_train, y_val = y_train_smote[val_samples:], y_train_smote[:val_samples]

# Lists to store training and validation losses
training_loss = []
validation_loss = []

# Train the model
  # You can adjust the number of epochs
for epoch in range(1, 51):
    model.fit(X_train, y_train)  # Fit for one epoch
    
    train_preds = model.predict_proba(X_train)
    train_loss = log_loss(y_train, train_preds)
    training_loss.append(train_loss)
    
    val_preds = model.predict_proba(X_val)
    val_loss = log_loss(y_val, val_preds)
    validation_loss.append(val_loss)

# Create a plot to visualize the training and validation loss
plt.plot(np.arange(1, len(training_loss) + 1), training_loss, label='Training Loss')
plt.plot(np.arange(1, len(validation_loss) + 1), validation_loss, label='Validation Loss')
plt.title('Training and Validation Loss for Neural Network')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


# ROC for the other models after feature selection 


# In[60]:


# Encode target labels into numeric format
label_encoder = LabelEncoder()
y_train_numeric = label_encoder.fit_transform(y_train_smote)
y_test_numeric = label_encoder.transform(y_test_smote)

# Define the selected features
selected_features = ['Curricular units 2nd sem (approved)', 'Curricular units 1st sem (approved)',
                     'Tuition fees up to date', 'Curricular units 2nd sem (grade)',
                     'Total Grade Average', 'Curricular units 1st sem (grade)']

# Select only the relevant columns from the training and test data
X_train_selected = X_train_smote[selected_features]
X_test_selected = X_test_smote[selected_features]

# Initialize classifiers
classifiers = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Gradient Boost': GradientBoostingClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Neural Network': MLPClassifier(max_iter=1000)
}

# Plot ROC curves for each classifier
plt.figure(figsize=(10, 8))
for name, classifier in classifiers.items():
    classifier.fit(X_train_selected, y_train_numeric)
    y_proba = classifier.predict_proba(X_test_selected)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_numeric, y_proba)
    auc = roc_auc_score(y_test_numeric, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




