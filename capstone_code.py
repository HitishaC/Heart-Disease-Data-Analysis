import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_theme(style="whitegrid")

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("=== DATASET OVERVIEW ===")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    return df

def clean_and_transform_data(df):
    null_count = df.isnull().sum().sum()
    print(f"\nMissing Values Found: {null_count}")
    if null_count > 0:
        df = df.dropna()

    initial_rows = len(df)
    df = df.drop_duplicates().copy()
    print(f"Duplicates Removed: {initial_rows - len(df)}")

    age_bins = [0, 40, 55, 100]
    age_labels = ['<40 (Young)', '40-55 (Middle-aged)', '>55 (Senior)']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    chol_bins = [0, 200, 239, 1000]
    chol_labels = ['Desirable (<200)', 'Borderline High (200-239)', 'High (>=240)']
    df['chol_category'] = pd.cut(df['chol'], bins=chol_bins, labels=chol_labels)

    return df

def plot_health_visualizations(df):
    plt.figure(figsize=(14, 10))


    plt.subplot(2, 2, 1)
    sns.countplot(x='target', hue='target', data=df, palette='Set2', legend=False)
    plt.title('1. Target Distribution (0 = Healthy, 1 = Heart Disease)', fontweight='bold')
    plt.xlabel('Target Outcome')
    plt.ylabel('Patient Count')
    plt.xticks([0, 1], ['No Disease', 'Heart Disease'])

    
    plt.subplot(2, 2, 2)
    sns.countplot(x='age_group', hue='target', data=df, palette='Set1')
    plt.title('2. Heart Disease Incidence Across Age Groups', fontweight='bold')
    plt.xlabel('Age Group')
    plt.ylabel('Patient Count')
    plt.legend(['No Disease', 'Heart Disease'])

    
    plt.subplot(2, 2, 3)
    sns.boxplot(x='target', y='thalach', hue='target', data=df, palette='Pastel1', legend=False)
    plt.title('3. Max Heart Rate Achieved (thalach) by Target', fontweight='bold')
    plt.xlabel('Target Outcome')
    plt.ylabel('Max Heart Rate (bpm)')
    plt.xticks([0, 1], ['No Disease', 'Heart Disease'])

   
    plt.subplot(2, 2, 4)
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
    sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('4. Feature Correlation Matrix', fontweight='bold')

    plt.tight_layout()
    plt.show()


df = load_data(r"c:\Users\HITISHA\OneDrive\Desktop\capstone\heart.csv")
df_cleaned = clean_and_transform_data(df)
plot_health_visualizations(df_cleaned)