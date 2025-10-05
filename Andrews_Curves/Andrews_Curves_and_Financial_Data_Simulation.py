####################################################################
# Andrews Curves and Financial Data Simulation
# Demonstrates how to create Andrews curves using simulated
# financial data with multiple categories.
####################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("Andrews Curves with Simulated Financial Data")
print("=" * 60)

# Simulate financial metrics fro companies
np.random.seed(123)
n_companies = 40

# Create three types of companies: growth, value, distressed
growth = np.random.randn(n_companies // 3, 5) * np.array([2, 1.5, 1, 0.5, 1]) + \
         np.array([15, 3, 25, 8, 12])
value = np.random.randn(n_companies // 3, 5) * np.array([1, 1, 0.8, 0.5, 1]) + \
        np.array([5, 8, 10, 15, 8])
distressed = np.random.randn(n_companies // 3 + 1, 5) * np.array([3, 2, 2, 1, 2]) + \
             np.array([-5, 1, 5, 3, 2])

financial_data = np.vstack([growth, value, distressed])
company_types = ['Growth'] * (n_companies // 3) + \
                ['Value'] * (n_companies // 3) + \
                ['Distressed'] * (n_companies // 3 + 1)

financial_df = pd.DataFrame(
    financial_data, 
    columns=['ROE', 'P/E_Ratio', 'Revenue_Growth', 'Dividend_Yield' ,'Debt_Ratio']
)
financial_df['Company_Type'] = company_types

# Standardize the features (important for Andrews curves)
scaler = StandardScaler()
financial_df_scaled = pd.DataFrame(
    scaler.fit_transform(financial_df.iloc[:, :-1]), 
    columns=financial_df.columns[:-1]
)
financial_df_scaled['Company_Type'] = company_types

# Create Andrews curves plot
plt.figure(figsize=(12, 6))
andrews_curves(financial_df_scaled, 'Company_Type', alpha=0.5, colormap='rainbow')
plt.title("Andrews Curves for Simulated Financial Data", fontsize=14, fontweight='bold')
plt.xlabel("t")
plt.ylabel("Andrews Function f(t)")
plt.legend(loc='best', title='Company Type')
plt.grid(True, alpha=0.3)
plt.show()