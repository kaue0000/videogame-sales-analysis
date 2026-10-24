ATENÇÃO: ISTO NÃO DEVE SER INTERPRETADO COMO UMA FORMA DE SOBREPOR O PDF ESCRITO. FOI REDIGIDO PURAMENTE PARA O ANDAMENTO DO PROJETO.

# Análise de vendas de videogames: Ocidente × Oriente

## Equipe
- Ozaeltton Matheus da Silva Medeiros (20231038060022)
- Kauê Adriano dos Santos Dantas (20241038060007)

**Repositório:** https://github.com/kaue0000/steam-sales-analysis

## Justificativa e problemática

Escolhemos o conjunto de dados **Video Game Sales**, que reúne vendas de videogames em diferentes plataformas (consoles, portáteis e PC) e as divide por região do mundo. Essa divisão permite comparar mercados com culturas e hábitos de consumo distintos.

**Problemática:** como o perfil dos jogos mais vendidos, em termos de gênero, plataforma e publisher, difere entre o Ocidente e o Oriente?

Para responder a isso, usamos como referência as três regiões em que o dataset divide as vendas: América do Norte, Europa e Japão. A partir delas, fazemos uma generalização em dois grandes grupos: **Ocidente** (América do Norte e Europa) e **Oriente** (Japão). Essa é uma simplificação assumida, já que o Japão não representa todo o Oriente e a coluna `Other_Sales` agrupa os demais países sem distinção.

Como o dataset não possui variáveis como preço, avaliações da crítica ou orçamento, o projeto tem caráter **descritivo e comparativo**: buscamos identificar padrões e diferenças de preferência entre as regiões, sem afirmar relações de causa e efeito sobre o sucesso de vendas.

## Conjunto de dados: Video Game Sales

### Visão geral
O conjunto **Video Game Sales** reúne vendas de videogames com mais de 100 mil cópias vendidas. Os dados foram obtidos por web scraping do site [vgchartz.com](https://www.vgchartz.com) e publicados no Kaggle por Gregory Smith (`gregorut`).

- **Fonte:** https://www.kaggle.com/datasets/gregorut/videogamesales
- **Script de coleta:** https://github.com/GregorUT/vgchartzScrape (Python com BeautifulSoup)
- **Arquivo:** `vgsales.csv` (~1,4 MB)
- **Tamanho:** 16.598 registros e 11 colunas (2 registros foram descartados na coleta por informação incompleta)
- **Cobertura temporal:** jogos lançados de 1980 a 2020, aproximadamente
- **Unidade de observação:** cada linha é um jogo em uma plataforma específica. Um mesmo título lançado em várias plataformas aparece em várias linhas.
- **Licença:** *(confira na página do Kaggle e preencha aqui)*

### Dicionário de variáveis

| Variável | Tipo | Descrição |
|---|---|---|
| `Rank` | Inteiro | Posição no ranking de vendas globais |
| `Name` | Texto | Nome do jogo |
| `Platform` | Categórica | Plataforma de lançamento (ex.: PC, PS4, Wii) |
| `Year` | Numérica | Ano de lançamento |
| `Genre` | Categórica | Gênero do jogo |
| `Publisher` | Categórica | Empresa publicadora |
| `NA_Sales` | Numérica | Vendas na América do Norte (em milhões de unidades) |
| `EU_Sales` | Numérica | Vendas na Europa (em milhões de unidades) |
| `JP_Sales` | Numérica | Vendas no Japão (em milhões de unidades) |
| `Other_Sales` | Numérica | Vendas no resto do mundo (em milhões de unidades) |
| `Global_Sales` | Numérica | Vendas totais no mundo (em milhões de unidades) |

### Observações e limitações
- A coluna `Year` contém valores ausentes ou incompletos, que precisam de tratamento antes da análise (a ausência também pode ocorrer em `Publisher`).
- Só entram jogos com vendas acima de 100 mil cópias, então o conjunto não representa todo o mercado.
- Como os dados vêm de scraping de um único site, podem refletir as estimativas e os vieses do VGChartz.
- As vendas são por plataforma. Para analisar um título como um todo, é preciso agrupar por `Name`.
- Os dados cobrem principalmente jogos até cerca de 2016, com poucos registros posteriores.