import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import yfinance as yf

# Definindo o ticker da criptomoeda e a data
ticker = "BTC-USD"  # Exemplo: Bitcoin
data_inicio = "2021-06-22"  # Data inicial (formato YYYY-MM-DD)

# Download de dados históricos
dados = yf.download(ticker, start=data_inicio, end=pd.Timestamp.today())

# Selecionando apenas a coluna 'Close' (fechamento)
fechamento = dados['Close']

# Cálculo das médias móveis exponenciais (EMA) de 20 e 50 períodos **dentro do DataFrame**
dados['ema_20'] = dados['Close'].ewm(span=20, min_periods=20).mean()
dados['ema_50'] = dados['Close'].ewm(span=50, min_periods=50).mean()

# Criando indicadores de compra e venda
sinais = []
for i in range(1, len(fechamento)):
    if dados['ema_20'][i] > dados['ema_50'][i] and dados['ema_20'][i - 1] <= dados['ema_50'][i - 1]:
        sinais.append(1)  # Compra
        print(f"Cruzamento de Compra em {dados.index[i].date()} ao preço de {fechamento[i]:.2f}")
    elif dados['ema_20'][i] < dados['ema_50'][i] and dados['ema_20'][i - 1] >= dados['ema_50'][i - 1]:
        sinais.append(-1)  # Venda
        print(f"Cruzamento de Venda em {dados.index[i].date()} ao preço de {fechamento[i]:.2f}")
    else:
        sinais.append(0)  # Neutro

# Adicionando sinal inicial para alinhar comprimentos
sinais.insert(0, 0)

# Treinando a árvore de decisão
X = dados[['ema_20', 'ema_50']].dropna()  # Entradas (EMA 20 e EMA 50)
y = sinais[-len(X):]  # Saída (sinais de compra/venda), alinhando o comprimento

modelo = DecisionTreeClassifier()
modelo.fit(X, y)

# Fazendo previsões para novos dados
novos_dados = {'ema_20': [dados['ema_20'].iloc[-1]], 'ema_50': [dados['ema_50'].iloc[-1]]}  # Dados para previsão
novos_dados = pd.DataFrame(novos_dados)

previsoes = modelo.predict(novos_dados)

# Interpretando a previsão
if previsoes[0] == 1:
    print("Sinal de compra!")
elif previsoes[0] == -1:
    print("Sinal de venda!")
else:
    print("Neutro.")

# Plotando gráficos
plt.figure(figsize=(12, 6))

plt.plot(fechamento, label='Preço')
plt.plot(dados['ema_20'], label='EMA 20')
plt.plot(dados['ema_50'], label='EMA 50')

# Destacando os sinais de compra e venda no gráfico
for i in range(len(sinais)):
    if sinais[i] == 1:
        plt.scatter(dados.index[i], fechamento[i], color='green', marker='^', label='Compra' if i == sinais.index(1) else "")
    elif sinais[i] == -1:
        plt.scatter(dados.index[i], fechamento[i], color='red', marker='v', label='Venda' if i == sinais.index(-1) else "")

plt.legend()
plt.title('Análise de Criptomoeda com Médias Móveis e Árvore de Decisão')
plt.show()
