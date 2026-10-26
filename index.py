import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Parâmetros da variável-alvo 
# Participação do Japão nas vendas (NA + EU + JP) de cada jogo:
# abaixo de LIMIAR_OCIDENTE -> "Ocidente"
# acima de LIMIAR_ORIENTE   -> "Oriente"
# entre os dois             -> "Equilibrado"
LIMIAR_OCIDENTE = 0.10
LIMIAR_ORIENTE = 0.50

data = pd.read_csv('./vgsales.csv')

print("-" * 20 + "Primeiras linhas do dataset" + "-" * 20)
print(data.head())

print("-" * 20 + "Últimas linhas do dataset" + "-" * 20)
print(data.tail())

print("-" * 20 + "Formato do dataset" + "-" * 20)
print(data.shape)  # esperado: cerca de 16.598 linhas e 11 colunas

print("-" * 20 + "Tipos das colunas" + "-" * 20)
print(data.dtypes)

# Valores nulos por coluna (o pandas já lê "N/A" do CSV como nulo).
# Publisher e Year possuem nulos; as colunas de vendas não.
print("-" * 20 + "Valores nulos por coluna" + "-" * 20)
print(data.isnull().sum())

# Linhas duplicadas
print("-" * 20 + "Linhas duplicadas" + "-" * 20)
print(data.duplicated().sum())

# Os valores 0 nas colunas de vendas têm significado para a análise (o jogo
# vendeu menos de 10 mil cópias na região, no arredondamento do dataset),
# portanto não são apagados nem substituídos.

# Exclusão de colunas desnecessárias para a análise:
# Rank: é calculado a partir de Global_Sales, que contém as vendas das
# regiões usadas para criar o alvo (vazamento de informação).
# Global_Sales: soma das regiões, incluindo as que usamos para o alvo.
# Other_Sales: fora do recorte Ocidente x Oriente.
# Name: identificador (milhares de valores únicos), sem poder de generalização.
# A coluna Year é MANTIDA: plataformas e gêneros mudam muito com a época,
# e o ano ajuda a separar o efeito temporal do efeito cultural.
data = data.drop(columns=['Rank', 'Name', 'Other_Sales', 'Global_Sales'])

# Ocidente = América do Norte + Europa; Oriente = Japão.
data["West_Sales"] = data["NA_Sales"] + data["EU_Sales"]
data["East_Sales"] = data["JP_Sales"]
total_regioes = data["West_Sales"] + data["East_Sales"]

# Jogos sem vendas nas três regiões (vendas apenas em Other_Sales) não têm
# como receber uma classe e são removidos.
sem_vendas = total_regioes == 0
print("-" * 20 + "Jogos removidos (sem vendas em NA, EU e JP)" + "-" * 20)
print(int(sem_vendas.sum()))
data = data[~sem_vendas].copy()
total_regioes = total_regioes[~sem_vendas]

# Comparar West_Sales com East_Sales diretamente favoreceria o Ocidente por
# construção (duas regiões somadas contra uma). Por isso o alvo usa a
# participação do Japão nas vendas de cada jogo: mede se o jogo performa de
# forma relativamente melhor no Oriente, independentemente do tamanho do mercado.
data["East_participation"] = data["East_Sales"] / total_regioes

# Construção da variável-alvo (Market) a partir da participação do Japão nas vendas.
# Indica qual mercado o jogo vende mais: Ocidente, Oriente ou Equilibrado.
data["Market"] = np.select(
    [
        data["East_participation"] < LIMIAR_OCIDENTE,
        data["East_participation"] > LIMIAR_ORIENTE,
    ],
    ["West", "East"],
    default="Balanced",
)

print("-" * 20 + "Distribuição da variável-alvo (Mercado)" + "-" * 20)
print(data["Market"].value_counts())
print(data["Market"].value_counts(normalize=True).round(3))
# Classes desbalanceadas: nas próximas etapas usar validação cruzada
# estratificada e métricas como F1 macro / acurácia balanceada.

# As colunas de vendas e derivadas saem das características, evitando vazamento de informação.
colunas_vendas = ["NA_Sales", "EU_Sales", "JP_Sales",
                  "West_Sales", "East_Sales", "East_participation"]

y = data["Market"]
X = data.drop(columns=colunas_vendas + ["Market"])

colunas_categoricas = ['Platform', 'Genre', 'Publisher']
colunas_numericas = ['Year']

# Pré-processamento abaixo
# Categóricas: nulos viram "Unknown"; categorias raras (poucos jogos, comum em
# Publisher) são agrupadas em uma categoria "infrequente" para evitar centenas
# de colunas quase vazias.
pipeline_categorico = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10)),
])

# Numéricas: nulos viram a mediana; depois padronização.
pipeline_numerico = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# Este "preprocessor" pode ser usado dentro de um Pipeline com o classificador
# nas próximas etapas, para ser reajustado a cada dobra da validação cruzada.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', pipeline_categorico, colunas_categoricas),
        ('num', pipeline_numerico, colunas_numericas),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_proc = preprocessor.fit_transform(X_train)  # aprende só no treino
X_test_proc = preprocessor.transform(X_test)        # apenas aplica no teste

print("-" * 20 + "Resultado do pré-processamento" + "-" * 20)
print("Treino:", X_train_proc.shape, "| Teste:", X_test_proc.shape)
print("Nº de colunas após codificação:", len(preprocessor.get_feature_names_out()))
