import sys
import os

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"ERROR: Missing required library. {e}")
    print("Please run this command in your terminal first: pip install pandas matplotlib numpy")
    input("\nPress Enter to close...")
    sys.exit()

if not os.path.exists('MATERN~1.CSV'):
    print("ERROR: Cannot find 'MATERN~1.CSV' in this folder!")
    print(f"Current folder contents: {os.listdir('.')}")
    input("\nPress Enter to close...")
    sys.exit()

print("="*60)
print("Step 1: Dataset loaded successfully!")
print("="*60)
df = pd.read_csv('MATERN~1.CSV')

print("\n--- Step 2: Display First 8 Rows ---")
print(df.head(8))
print("\n--- Step 2: Display First 12 Rows ---")
print(df.head(12))
print("\n--- Step 2: Display First 25 Rows ---")
print(df.head(25))
print("\n" + "="*50)

print("\n--- Step 3: Data Types of Elements in Individual Rows (First 5 Rows) ---")
print(df.iloc[:5].applymap(type))
print("\n" + "="*50)

print("\n--- Step 4: Data Type of Entire Columns ---")
print(df.dtypes)
print("\n" + "="*50)

print("\n--- Step 5: Column Names ---")
print(df.columns.tolist())
print("\n" + "="*50)

df_copy = df.copy()
for col in df_copy.columns:
    if pd.api.types.is_integer_dtype(df_copy[col]):
        df_copy[col] = df_copy[col].astype(float)
    elif pd.api.types.is_float_dtype(df_copy[col]):
        df_copy[col] = df_copy[col].round().astype('Int64')

print("\n--- Step 6: Converted New Data Frame Column Types ---")
print(df_copy.dtypes)
print("\nNew Data Frame Head Sample:")
print(df_copy.head(3))
print("\n" + "="*50)

df_non_numeric = df.select_dtypes(exclude=[np.number])
print("\n--- Step 7: Non-Numeric Data Frame (Head) ---")
print(df_non_numeric.head())
print("\n" + "="*50)

print("\n--- Step 8: Greater Than Filter (SystolicBP > 130) ---")
print(df[df['SystolicBP'] > 130].head(2))

print("\n--- Step 8: Greater Than or Equal To Filter (SystolicBP >= 130) ---")
print(df[df['SystolicBP'] >= 130].head(2))

print("\n--- Step 8: Less Than Filter (SystolicBP < 100) ---")
print(df[df['SystolicBP'] < 100].head(2))

print("\n--- Step 8: Less Than or Equal To Filter (SystolicBP <= 100) ---")
print(df[df['SystolicBP'] <= 100].head(2))

print("\n--- Step 8: Equal To Filter (RiskLevel == 'high risk') ---")
print(df[df['RiskLevel'] == 'high risk'].head(2))
print("\n" + "="*50)

print("\n--- Step 9: Grouping Kind 1 (Mean/Average Value by RiskLevel) ---")
print(df.groupby('RiskLevel').mean(numeric_only=True))

print("\n--- Step 9: Grouping Kind 2 (Maximum Value by RiskLevel) ---")
print(df.groupby('RiskLevel').max(numeric_only=True))

print("\n--- Step 9: Grouping Kind 3 (Minimum Value by RiskLevel) ---")
print(df.groupby('RiskLevel').min(numeric_only=True))

print("\n--- Step 9: Grouping Kind 4 (Row Entry Count by RiskLevel) ---")
print(df.groupby('RiskLevel').count())
print("\n" + "="*50)

print("\n--- Step 10: Specific Row Range (1-5) and Columns (1st, 2nd, 4th, 6th) ---")
print(df.iloc[0:5, [0, 1, 3, 5]])
print("\n" + "="*50)

print("\n--- Step 11: Third Through Tenth Rows & First Four Columns ---")
print(df.iloc[2:10, 0:4])
print("\n" + "="*50)

print("\n--- Step 12: Custom Multi-Criteria Filtering (SystolicBP > 120 & High Risk) ---")
custom_filter = df[(df['SystolicBP'] > 120) & (df['RiskLevel'] == 'high risk')]
print(custom_filter.head())
print("\n" + "="*50)

print("\n--- Step 13: Slicing using Label Names (.loc) ---")
print(df.loc[5:10, ['Age', 'SystolicBP', 'RiskLevel']])
print("\n" + "="*50)

print("\n--- Step 14: Slicing using Index Integer Positions Only (.iloc) ---")
print(df.iloc[5:10, 0:3])
print("\n" + "="*50)

print("\n--- Step 15: Null Counts Separated By Column ---")
print(df.isnull().sum())
print("\n" + "="*50)

print("\n--- Step 16: Total Missing Values Over Whole Dataset ---")
print(df.isnull().sum().sum())
print("\n" + "="*50)

missing_cols = df.columns[df.isnull().any()].tolist()
print(f"\nStep 17: Identified Columns with Missing Data: {missing_cols}")

df_mean_filled = df.copy()
df_mean_filled[missing_cols] = df_mean_filled[missing_cols].fillna(df_mean_filled[missing_cols].mean())
print("\n--- Step 17a: Missing Data Handled with Column Mean (Head) ---")
print(df_mean_filled.head(3))

df_min_filled = df.copy()
df_min_filled[missing_cols] = df_min_filled[missing_cols].fillna(df_min_filled[missing_cols].min())

df_max_filled = df.copy()
df_max_filled[missing_cols] = df_max_filled[missing_cols].fillna(df_max_filled[missing_cols].max())
print("\n" + "="*50)

print("\n--- Step 18: Exporting Clean Data ---")
df_mean_filled.to_csv('matern_health_ARLEELEMPAN.csv', index=False)
print("File exported successfully as 'matern_health_ARLEELEMPAN.csv'!")
print("\n" + "="*50)

print("\n--- Step 19: Generating Visualizations using Matplotlib ---")

fig, ax = plt.subplots()
ax.hist(df_mean_filled['SystolicBP'], bins=15, color='skyblue', edgecolor='black')
ax.set_title('Distribution of Systolic Blood Pressure')
ax.set_xlabel('Systolic Blood Pressure (mmHg)')
ax.set_ylabel('Number of Patients (Frequency)')
plt.tight_layout()
plt.savefig('histogram_systolic_bp.png')
plt.close()
print("-> Saved: histogram_systolic_bp.png")

risk_counts = df_mean_filled['RiskLevel'].value_counts()
fig, ax = plt.subplots()
ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
ax.set_title('Proportions of Risk Levels')
plt.tight_layout()
plt.savefig('pie_chart_risk_level.png')
plt.close()
print("-> Saved: pie_chart_risk_level.png")

avg_bs = df_mean_filled.groupby('RiskLevel')['BS'].mean().sort_values()
fig, ax = plt.subplots()
avg_bs.plot(kind='bar', color='salmon', edgecolor='black', ax=ax)
ax.set_title('Average Blood Sugar by Risk Level')
ax.set_xlabel('Risk Level')
ax.set_ylabel('Average Blood Sugar (BS)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('bar_chart_blood_sugar.png')
plt.close()
print("-> Saved: bar_chart_blood_sugar.png")

avg_sys_by_age = df_mean_filled.groupby('Age')['SystolicBP'].mean().sort_index()
fig, ax = plt.subplots()
ax.plot(avg_sys_by_age.index, avg_sys_by_age.values, marker='o', color='purple', linestyle='-')
ax.set_title('Average Systolic Blood Pressure by Age')
ax.set_xlabel('Age of Patient')
ax.set_ylabel('Average SystolicBP')
plt.tight_layout()
plt.savefig('line_graph_systolic_by_age.png')
plt.close()
print("-> Saved: line_graph_systolic_by_age.png")

fig, ax = plt.subplots()
df_mean_filled.boxplot(column='BodyTemp', by='RiskLevel', grid=False, ax=ax)
ax.set_title('Body Temperature Distribution by Risk Level')
plt.suptitle('') 
ax.set_xlabel('Risk Level')
ax.set_ylabel('Body Temperature')
plt.tight_layout()
plt.savefig('boxplot_body_temp.png')
plt.close()
print("-> Saved: boxplot_body_temp.png")

print("\n" + "="*60)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("="*60)

input("\nAll steps finished! Press Enter to close this window...")