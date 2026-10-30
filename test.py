import pandas as pd

df = pd.read_csv("uploads/Telco dataset-26.csv")  # adjust filename

col = "MonthlyCharges"
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

print(f"Min: {df[col].min()}")
print(f"Max: {df[col].max()}")
print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Lower (3x): {Q1 - 3.0 * IQR}")
print(f"Upper (3x): {Q3 + 3.0 * IQR}")
print(f"Total values above upper: {(df[col] > Q3 + 3.0 * IQR).sum()}")
print(f"Total values below lower: {(df[col] < Q1 - 3.0 * IQR).sum()}")

lower_p = df[col].quantile(0.01)
upper_p = df[col].quantile(0.99)
print(f"Percentile Lower (1%): {lower_p}")
print(f"Percentile Upper (99%): {upper_p}")
print(f"Values above 99th: {(df[col] > upper_p).sum()}")
print(f"Values below 1st: {(df[col] < lower_p).sum()}")

import numpy as np
mean = df[col].mean()
std = df[col].std()
lower_z = mean - 3 * std
upper_z = mean + 3 * std
print(f"Mean: {mean}")
print(f"Std: {std}")
print(f"Z-Score Lower: {lower_z}")
print(f"Z-Score Upper: {upper_z}")
print(f"Values above upper: {(df[col] > upper_z).sum()}")
print(f"Values below lower: {(df[col] < lower_z).sum()}")