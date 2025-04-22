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

Tabela/resumo rápido:

Melhor acurácia: KNN (~X%)

Pior desempenho: Naïve Bayes (~Y%), devido à suposição de independência.

Principais erros: tendência de confundir “Normal Weight” com “Overweight Level I” em todos os modelos.

## Interpretabilidade Modelo a Modelo 

Decision Tree

Mostre o grafo simplificado (ou destaque nós).

Features mais importantes no topo (peso, altura, idade).

Naïve Bayes

Explique as probabilidades condicionais de duas ou três features chave.

Limitação: não capta correlações (dificulta separar classes próximas).

KNN + SHAP/LIME

Demonstre um exemplo de explicação com SHAP (gráfico de barras para uma instância).

A dificuldade “pura” do KNN e como SHAP ajuda a entender quais pontos vizinhos influenciam mais.

## Comparação Geral e Lições 

Facilidade de interpretação:

Árvore (visual, regras),

Naïve Bayes (fácil ver probabilidades),

KNN (sem explicação natural, depende de pós‑análise).

Limitações:

Viéses inesperados (ex.: influência de gênero).

Trade‑off entre performance e transparência.

## Conclusões e Próximos Passos (1min)

Conclusões principais:

Modelos simples oferecem certa transparência, mas nada substitui análises com SHAP/LIME em casos “black‑box”.

KNN foi mais acurado, Decision Tree mais intuitivo, NB o mais rápido porém menos preciso.


8. Encerramento 

Agradecimento: “Obrigado pela atenção! Mais detalhes no código e no README.”


