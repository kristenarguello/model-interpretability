Roteiro para video de apresentacao t1 Machine Learning

## Abertura

Apresentar grupo e trbalho 

Objetivo geral: “O objetivo é comparar a interpretabilidade de KNN, Naïve Bayes e Árvore de Decisão num caso real de classificação de obesidade.”

## Descrição do Dataset e Problema 

Origem: “Escolhemos o dataset ‘Estimation of Obesity Levels…’ do UCI (544 instâncias, 17 features mistas).”

Variáveis principais: numéricas (altura, peso, idade…) e categóricas (gênero, hábitos alimentares, etc.).

Alvo: 6 classes de obesidade, de Insufficient Weight a Obesity Type III.

Motivação: “Temos pelo menos 5 features e um problema multiclasses onde interpretabilidade é crítica.”

## Pré‑processamento e Pipeline de Treinamento 

Limpeza e codificação:

Tratamento de missing (nenhum no dataset)

Normalização de numéricas (MinMax ou z‑score)

One‑hot/Ordinal encoding de categóricas.

Divisão treino/teste (80/20).

Modelos treinados:

KNN (k otimizado com cross‑validation)

Gaussian Naïve Bayes

Decision Tree (profundidade limitada para evitar overfitting)

Métricas usadas: acurácia, precisão, recall, F1‑score e matriz de confusão.

## Resultados de Desempenho 

Melhor acurácia: KNN (~X%)

Pior desempenho: Naïve Bayes (~Y%), devido à suposição de independência.

Principais erros: tendência de confundir “Normal Weight” com “Overweight Level I” em todos os modelos.

## Interpretabilidade Modelo a Modelo 

### Decision Tree
“Decision Tree – Matriz de Confusão”

- Mostrar a heatmap da confusão.


“Na diagonal vemos que o modelo acerta quase 100% em Obesity_Type_III e ~87% em Insufficient_Weight.
Porém, classes intermediárias como Normal_Weight (56% de recall) e Overweight_Level_I (67%) são frequentemente confundidas umas com as outras.”

“Decision Tree – SHAP Summary Plot”

- Mostrar o gráfico de barras de SHAP.

“Aqui, gender_Male, age e height são as três features de maior impacto médio.
Gênero não faz parte do IMC, mas o modelo usa esse viés populacional para diferenciar instâncias quando peso e altura não bastam.”


### Naïve Bayes
“Naïve Bayes – Matriz de Confusão”

- Mostrar a heatmap.

“NB erra muito nas classes do meio: Normal_Weight acerta só ~24% e Overweight_Level_I ~26%.
A maioria das instâncias acaba sendo ‘empurrada’ para Obesity_Type_II, devido à suposição de independência.”

“Naïve Bayes – SHAP Summary Plot”

- Mostrar barras SHAP.

“Gênero (Male/Female) e histórico familiar aparecem no topo, enquanto height quase some.
Isso mostra que o NB favorece categorias binárias quando as distribuições contínuas se sobrepõem.”


### K‑Nearest Neighbors (KNN)
“KNN – Matriz de Confusão”

- Mostrar a heatmap.

“KNN funciona muito bem nos extremos (Insufficient e Obesity_Type_III com >87% de acerto), mas ainda confunde classes intermediárias: Normal_Weight e Overweight_Level_I.”

“KNN – SHAP Summary Plot”

- Mostrar barras SHAP.

“height e age continuam sendo as top features, seguidas por frequência de vegetais e atividade física.
SHAP nos permite entender quais variáveis influenciam cada previsão, algo que o KNN puro não oferece.”


## Conclusões e Próximos Passos (1min)

Conclusões principais:

Modelos simples oferecem certa transparência, mas nada substitui análises com SHAP/LIME em casos “black‑box”.

KNN foi mais acurado, Decision Tree mais intuitivo, NB o mais rápido porém menos preciso.


8. Encerramento 

Agradecimento: “Obrigado pela atenção! Mais detalhes no código e na documentacao”


