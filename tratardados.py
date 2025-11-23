import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt


arquivo = r"creditcard.csv"
dados = pd.read_csv(arquivo)




dados = dados.dropna().drop_duplicates()



fraudes = dados[dados["Class"] == 1]
normais = dados[dados["Class"] == 0]

Q1 = normais.quantile(0.25)
Q3 = normais.quantile(0.75)
IQR = Q3 - Q1


normais_sem_outliers = normais[
    ~((normais < (Q1 - 1.5 * IQR)) | (normais > (Q3 + 1.5 * IQR))).any(axis=1)
]


dados_sem_outliers = pd.concat([normais_sem_outliers, fraudes], ignore_index=True)
print(f"Linhas após remover outliers (mantendo fraudes): {len(dados_sem_outliers)}")


scaler = StandardScaler()
colunas_numericas = dados_sem_outliers.drop(columns=["Class"]).columns


dados_sem_outliers.loc[:, colunas_numericas] = scaler.fit_transform(
    dados_sem_outliers[colunas_numericas]
)


corr = dados_sem_outliers.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Matriz de Correlação")
plt.show()


correladas = [
    (x, y)
    for x in corr.columns
    for y in corr.columns
    if x != y and abs(corr.loc[x, y]) > 0.9
]
print(f"Variáveis com alta correlação: {correladas}")


X = dados_sem_outliers.drop(columns=["Class"])
y = dados_sem_outliers["Class"]

print("\nDistribuição original das classes:")
print(y.value_counts())


if len(y.unique()) > 1:
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("\nDistribuição após balanceamento:")
    print(pd.Series(y_res).value_counts())
else:
    
    X_res, y_res = X, y


X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
)

print(f"\nTamanho do treino: {len(X_train)}")
print(f"Tamanho do teste:  {len(X_test)}")

print("\nPrimeiras linhas da base tratada e balanceada:")
print(pd.concat([X_train, y_train], axis=1).head())
