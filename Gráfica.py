import matplotlib.pyplot as plt
import numpy as np

# Data
years = [2011, 2014, 2017, 2021]

financial_inclusion_mexico = [27, 39, 37, 49]  # Percentage of population with financial accounts in Mexico

financial_inclusion_india = [35, 53, 80, 78]  # Percentage of population with financial accounts in India

financial_inclusion_china = [63, 79, 80, 89]  # Percentage of population with financial accounts in China

financial_inclusion_argentina = [56, 68, 70, 84]  # Percentage of population with financial accounts in Brazil

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Mexico plot
axs[0, 0].set_title('Mexico')
axs[0, 0].plot(years, financial_inclusion_mexico, 'o-', color='green', label='Cuentas Financieras')
axs[0, 0].set_xlabel('Año')
axs[0, 0].set_ylabel('%')
axs[0, 0].legend()

# India plot
axs[0, 1].set_title('India')
axs[0, 1].plot(years, financial_inclusion_india, 'x-', color='lightblue', label='Cuentas Financieras')
axs[0, 1].set_xlabel('Año')
axs[0, 1].set_ylabel('%')
axs[0, 1].legend()

# China plot
axs[1, 0].set_title('China')
axs[1, 0].plot(years, financial_inclusion_china, '^-', color='lightcoral', label='Cuentas Financieras')
axs[1, 0].set_xlabel('Año')
axs[1, 0].set_ylabel('%')
axs[1, 0].legend()

# Argentina plot
axs[1, 1].set_title('Argentina')
axs[1, 1].plot(years, financial_inclusion_argentina, 'h-', color='mediumpurple', label='Cuentas Financieras')
axs[1, 1].set_xlabel('Año')
axs[1, 1].set_ylabel('%')
axs[1, 1].legend()

fig.tight_layout()
plt.show()