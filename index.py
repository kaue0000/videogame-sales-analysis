import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

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

# Rank: é calculado a partir de Global_Sales, que contém as vendas das
# regiões-alvo (vazamento de informação).
# Global_Sales: soma das regiões, incluindo as que usamos como alvo.
# Other_Sales: fora do recorte Ocidente x Oriente. Inclui vendas de países que não são alvo da análise, portanto não é útil para o modelo.
# Name: identificador (milhares de valores únicos), sem poder de generalização.
# A coluna Year é MANTIDA: plataformas e gêneros mudam muito com a época,
# e o ano ajuda a separar o efeito temporal do efeito cultural.
data = data.drop(columns=['Rank', 'Name', 'Other_Sales', 'Global_Sales'])

# Construção das variáveis-alvo (Ocidente e Oriente)
data["West_Sales"] = data["NA_Sales"] + data["EU_Sales"]
data["East_Sales"] = data["JP_Sales"]

alvos = ["West_Sales", "East_Sales"]
colunas_vendas = ["NA_Sales", "EU_Sales", "JP_Sales"]

y = data[alvos]
X = data.drop(columns=colunas_vendas + alvos)

colunas_categoricas = ['Platform', 'Genre', 'Publisher']
colunas_numericas = ['Year']

# Pipelines de pré-processamento
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

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', pipeline_categorico, colunas_categoricas),
        ('num', pipeline_numerico, colunas_numericas),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# aprende só no treino
X_train_proc = preprocessor.fit_transform(X_train)
# apenas aplica no teste
X_test_proc = preprocessor.transform(X_test)

print("-" * 20 + "Resultado do pré-processamento" + "-" * 20)
print("Treino:", X_train_proc.shape, "| Teste:", X_test_proc.shape)
print("Nº de colunas após codificação:", len(preprocessor.get_feature_names_out()))