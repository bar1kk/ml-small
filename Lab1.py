import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

file_path = './data/variant_13_Male.csv'
df = pd.read_csv(file_path)
imputer = KNNImputer(n_neighbors=5)

print("1. Початковий аналіз:")
print(f"Розмір набору даних: {df.shape}")
print("\nПерші 10 рядків:")
print(df.head(10))
print("\nІнформація про типи даних:")
df.info()
print("\nОцінка набору даних:")
print(df.describe(include='all'))

print("\n2. Пошук пропущених та аномальних значень:")
print("Кількість пропущених значень у кожній колонці:")
print(df.isna().sum())

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

df[num_cols] = imputer.fit_transform(df[num_cols])

for col in cat_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nПропущені значення після обробки:")
print(df.isna().sum())

anomaly_report = []
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    if count > 0:
        df[col] = df[col].clip(lower_bound, upper_bound)
        anomaly_report.append({'Показник': col, 'Кількість аномалій': count})


df_anomalies = pd.DataFrame(anomaly_report)
print("\nЗнайдені аномалії:")
print(df_anomalies.to_string(index=False))

print("\n3. Створення нової ознаки - Waist_Height_Ratio")
df['Waist_Height_Ratio'] = df['Waist_Circumference'] / df['Height']
print(df[['Waist_Circumference', 'Height', 'Waist_Height_Ratio']].head(10))
df = df.drop(columns=['Waist_Circumference', 'Height', 'Weight'])

print("\n4. Теплова карта кореляції")
target = 'BMI'
num_cols_for_corr = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(12, 10))
sns.heatmap(df[num_cols_for_corr].corr(), annot=False, cmap='coolwarm')
plt.title("Теплова карта кореляції всіх числових ознак")
plt.show()

print("\n5. Відбір ознак за порогом кореляції та перевірка значущості")
corr_matrix = df[num_cols_for_corr].corr()
corr_with_target = corr_matrix[target].drop(target)

threshold = 0.3
good_corr_features = corr_with_target[corr_with_target.abs() > threshold].index.tolist()
print("Ознаки, які мають кореляцію з цільовою змінною вище порогу (|r| > 0.3):")
print(good_corr_features)

corr_results = []
significant_features = []

print("\nСтатистична значущість кореляції для відібраних ознак:")
for col in good_corr_features:
    r_p, p_p = pearsonr(df[col], df[target])
    r_s, p_s = spearmanr(df[col], df[target])
    
    if p_p < 0.05 and p_s < 0.05:
        significant_features.append(col)

    corr_results.append({
        'Ознака': col,
        'Пірсон r': r_p,
        'p-value (Пірсон)': p_p,
        'Значущість (Пірсон)': p_p < 0.05,
        'Спірмен ρ': r_s,
        'p-value (Спірмен)': p_s,
        'Значущість (Спірмен)': p_s < 0.05
    })

corr_df = pd.DataFrame(corr_results)
pd.set_option('display.colheader_justify', 'center')
print(corr_df.to_string(index=False, justify='center', col_space=10))

print("\n6. Масштабування значущих ознак та цільової змінної")
scaler = RobustScaler()
cols_to_scale = significant_features + [target]

df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
df_final = df_scaled[cols_to_scale]

print(df_final.describe().to_string())

output_file = 'processed_variant_13.csv'
df_final.to_csv(output_file, index=False)
print(f"\nОброблений набір даних збережено у файл: {output_file}")