import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Baixe os dados históricos do Ethereum (ETH) desde 2019 até hoje
eth_data = yf.download('ETH-USD', start='2019-01-01', end='2024-05-28', progress=False)

# Calcule as médias móveis de 20 e 50 dias
eth_data['SMA20'] = eth_data['Close'].rolling(window=20).mean()
eth_data['SMA50'] = eth_data['Close'].rolling(window=50).mean()

# Sinal de compra: quando a média móvel de 20 cruza acima da média móvel de 50
eth_data['Signal'] = np.where(eth_data['SMA20'] > eth_data['SMA50'], 1, 0)

# Sinal de venda: quando a média móvel de 50 cruza abaixo da média móvel de 20
eth_data['Signal'] = np.where(eth_data['SMA50'] > eth_data['SMA20'], -1, eth_data['Signal'])

# Filtrar os pontos de cruzamento
crossings = eth_data[eth_data['Signal'].diff() != 0]

# Exibir as datas em que as médias se cruzaram
print("Datas de cruzamento:")
for date, row in crossings.iterrows():
    print(f"{date.date()}: {'Compra' if row['Signal'] == 1 else 'Venda'}")

# Plotar o gráfico com as médias móveis
plt.figure(figsize=(12, 6))
plt.plot(eth_data.index, eth_data['Close'], label='Preço de fechamento')
plt.plot(eth_data.index, eth_data['SMA20'], label='Média móvel de 20 dias', linestyle='--')
plt.plot(eth_data.index, eth_data['SMA50'], label='Média móvel de 50 dias', linestyle='--')
plt.scatter(crossings.index, crossings['Close'], marker='o', color='red', label='Cruzamento')
plt.title('Análise de Compra e Venda de Ethereum com Médias Móveis')
plt.xlabel('Data')
plt.ylabel('Preço (USD)')
plt.legend()
plt.show()
