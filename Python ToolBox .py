# ============================================
# STEP 1: DATA LOADING
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/rajanand/Desktop/Python /assembly-elections-data-at-candidate-level.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.shape)

# ============================================
# STEP 2: DATA CLEANING
# ============================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df['sex'].fillna(df['sex'].mode()[0], inplace=True)
df['age'].fillna(df['age'].mean(), inplace=True)
df['candidate_type'].fillna("Unknown", inplace=True)

df['total_electors'].fillna(df['total_electors'].mean(), inplace=True)
df['vote_share_percentage'].fillna(df['vote_share_percentage'].mean(), inplace=True)
df['margin_percentage'].fillna(df['margin_percentage'].mean(), inplace=True)
df['turnout_percentage'].fillna(df['turnout_percentage'].mean(), inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# STEP 3: EDA + VISUALIZATION (Objective 3 & 4)
# ============================================
# Objective 1. TOP 10 PARTIES BY TOTAL VOTES
# ============================================

party_votes = df.groupby('party')['total_votes'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=party_votes.values, y=party_votes.index)

for i, v in enumerate(party_votes.values):
    plt.text(v, i, str(int(v)), va='center')

plt.title("Top 10 Parties by Total Votes")
plt.xlabel("Total Votes")
plt.ylabel("Party")
plt.show()

# ============================================
#  Objective 2. GENDER DISTRIBUTION OF CANDIDATE
# ============================================

gender_count = df['sex'].value_counts()

plt.figure(figsize=(7,7))
plt.pie(gender_count, labels=gender_count.index, autopct='%1.2f%%', startangle=90)

plt.title("Gender Distribution of Candidates")
plt.show()

# ============================================
# Objective 3. AGE DISTRIBUTION ANALYSIS
# ============================================

plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=20, kde=True)

plt.title("Age Distribution of Candidates")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ============================================
# Objective 4.  CORRELATION ANALYSIS (AGE VS VOTES)
# ============================================

corr = df[['age','total_votes']].corr()

print("\nCorrelation between Age and Total Votes:")
print(corr)

plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

# ============================================
# Objective 5.  STATE-WISE VOTE ANALYSIS
# ============================================

state_votes = df.groupby('state_name')['total_votes'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
ax = sns.barplot(x=state_votes.values, y=state_votes.index)
for i, v in enumerate(state_votes.values):
    ax.text(v, i, str(int(v)), va='center')

plt.title("Top 10 States by Total Votes")
plt.xlabel("Total Votes")
plt.ylabel("State")
plt.show()


# STEP 4: OUTLIER DETECTION
# ============================================
# OUTLIER DETECTION (AGE)
# ============================================

Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['age'] < lower) | (df['age'] > upper)]

print("Number of outliers in age:", len(outliers))

plt.figure(figsize=(6,4))
sns.boxplot(x=df['age'])
plt.title("Outlier Detection (Age)")
plt.show()


# STEP 5: NORMALIZATION
# ============================================
# NORMALIZATION
# ============================================

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[['age','total_votes']] = scaler.fit_transform(df[['age','total_votes']])

print(df[['age','total_votes']].head())

# STEP 6: MACHINE LEARNING (Linear Regression)
# ============================================
# LINEAR REGRESSION MODEL
# ============================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['age']]
y = df['total_votes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Regression Graph
# Sort for smooth line
sorted_idx = X_test['age'].argsort()
X_test_sorted = X_test.iloc[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.figure(figsize=(8,5))
plt.scatter(X_test['age'], y_test, alpha=0.3)
plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=3)

plt.title("Linear Regression: Age vs Total Votes")

plt.xlabel("Age")

plt.ylabel("Total Votes")

plt.show()


# STEP 7: MODEL EVALUATION
# ============================================
# MODEL EVALUATION
# ============================================

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)
print("MAE:", mae)







