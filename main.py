import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns


# Функция для преобразования и обработки данных акций
def process_stock_data(file_path):
    data = pd.read_csv(file_path)
    data['Дата'] = pd.to_datetime(data['Дата'], dayfirst=True)

    # Удаление точек в качестве разделителей тысяч и замена запятых на точки
    for column in ['Цена', 'Откр.', 'Макс.', 'Мин.']:
        data[column] = data[column].str.replace('.', '').str.replace(',', '.').astype(float)

    # Продолжение обработки данных
    filtered_data = data[data['Дата'] >= '2021-12-21'].copy()
    filtered_data['Еженедельное изменение'] = filtered_data['Цена'].pct_change(periods=7)
    return filtered_data


# Загрузка и обработка данных акций
gazp_data = process_stock_data('C:/Users/79377/Desktop/terverdata/Прошлые данные - GAZP.csv')
yndx_data = process_stock_data('C:/Users/79377/Desktop/terverdata/Прошлые данные - YNDX.csv')
tfnx_data = process_stock_data('C:/Users/79377/Desktop/terverdata/Прошлые данные по TFNX.csv')
lkoh_data = process_stock_data('C:/Users/79377/Desktop/terverdata/Прошлые данные - LKOH.csv')
ozondr_data = process_stock_data('C:/Users/79377/Desktop/terverdata/Прошлые данные - OZONDR.csv')

# Визуализация изменения цен акций
plt.figure(figsize=(15, 7))
plt.plot(gazp_data['Дата'], gazp_data['Цена'], label='GAZP')
plt.plot(yndx_data['Дата'], yndx_data['Цена'], label='YNDX')
plt.plot(tfnx_data['Дата'], tfnx_data['Цена'], label='TFNX')
plt.plot(lkoh_data['Дата'], lkoh_data['Цена'], label='LKOH')
plt.plot(ozondr_data['Дата'], ozondr_data['Цена'], label='OZONDR')
plt.title('Изменения цен акций')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.show()

# Расчет средней доходности и риска для всех акций
returns = pd.DataFrame({
    'GAZP': gazp_data['Цена'].pct_change(),
    'YNDX': yndx_data['Цена'].pct_change(),
    'TFNX': tfnx_data['Цена'].pct_change(),
    'LKOH': lkoh_data['Цена'].pct_change(),
    'OZONDR': ozondr_data['Цена'].pct_change()
})

mean_returns = returns.mean()
risk = returns.std()

# Визуализация средней доходности и риска
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Визуализация средней доходности
mean_returns.plot(kind='bar', ax=axes[0], color='blue')
axes[0].set_title('Средняя доходность')
axes[0].set_ylabel('Средняя доходность')

# Визуализация риска (волатильности)
risk.plot(kind='bar', ax=axes[1], color='orange')
axes[1].set_title('Риск (волатильность)')
axes[1].set_ylabel('Риск')

# Расчет матриц корреляций и ковариаций
correlation_matrix = returns.corr()
covariance_matrix = returns.cov()

# Создание графиков
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Тепловая карта матрицы корреляций
sns.heatmap(correlation_matrix, ax=ax[0], annot=True, cmap='coolwarm')
ax[0].set_title('Тепловая карта матрицы корреляций')

# Тепловая карта матрицы ковариаций
sns.heatmap(covariance_matrix, ax=ax[1], annot=True, cmap='coolwarm')
ax[1].set_title('Тепловая карта матрицы ковариаций')

plt.tight_layout()
plt.show()


# Функции для оптимизации портфеля (Модель Марковица)
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def minimize_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


# Ограничения для оптимизатора
num_assets = len(returns.columns)
args = (mean_returns, covariance_matrix)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0.0, 1.0) for asset in range(num_assets))

# Оптимизация портфеля
num_assets = len(returns.columns)
initial_guess = num_assets * [1. / num_assets]
optimal_portfolio = minimize(minimize_volatility, initial_guess, args=(mean_returns, covariance_matrix),
                             method='SLSQP', bounds=bounds, constraints=constraints)

# Визуализация оптимальных весов акций в портфеле с помощью круговой диаграммы
plt.figure(figsize=(6, 6))
optimal_weights = optimal_portfolio['x']
stock_labels = ['GAZP', 'YNDX', 'TFNX', 'LKOH', 'OZONDR']
plt.pie(optimal_weights, labels=stock_labels, autopct='%1.1f%%')
plt.title('Оптимальный вес акций в портфеле')
plt.show()
