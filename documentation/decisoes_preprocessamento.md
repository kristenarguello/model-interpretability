## Colunas de características (features)

- **gender**  
  categórica (nominal)  
  Valores: Masculino, Feminino  

- **age**  
  numérica (float)  
  Arredondada para o inteiro mais próximo para representar anos completos — evita sensibilidade desnecessária do modelo a decimais (ex: 20.7 vs 21.3).  
  Convertida para float para manter a consistência no tipo de dado.

- **height**  
  numérica (float)  
  Arredondada para 2 casas decimais para reduzir ruído e evitar granularidade excessiva.

- **weight**  
  numérica (float)  
  Arredondada para 2 casas decimais pelo mesmo motivo da altura — entrada mais limpa para o modelo.

- **family_history_overweight**  
  categórica (nominal)  
  Variável binária sim/não — convertida para tipo categórico.

- **high_caloric_food**  
  categórica (nominal)  
  Indica se a pessoa consome frequentemente alimentos altamente calóricos — tratada como categoria nominal.

- **freqof_vegetables**  
  categórica (ordinal)  
  Frequência do consumo de vegetais — possui uma ordem natural (ex: nunca < às vezes < sempre).

- **number_of_main_meals**  
  numérica (discreta, do tipo razão)  
  Número de refeições principais por dia — tratada como numérica já que a faixa é relativamente ampla e numericamente significativa.  
  Arredondada e armazenada como float.

- **food_between_meals**  
  categórica (ordinal)  
  Descreve a frequência de lanches entre as refeições — possui ordem natural (ex: nunca < às vezes < frequentemente).

- **smoking**  
  categórica (nominal)  
  Variável binária sim/não — tratada como categoria nominal.

- **water_intake**  
  categórica (ordinal)  
  Frequência de consumo diário de água — os valores têm uma progressão clara.

- **calories_monitoring**  
  categórica (nominal)  
  Variável binária que indica se a pessoa monitora a ingestão calórica — tratada como nominal.

- **freq_physical_activity**  
  categórica (ordinal)  
  Níveis ordenados de frequência de atividade física — valor maior = mais atividade.

- **time_using_technology**  
  categórica (ordinal)  
  Tempo gasto em dispositivos eletrônicos — mais tempo implica comportamento mais sedentário, portanto tratada como ordinal.

- **alcohol_consumption**  
  categórica (ordinal)  
  Frequência de consumo de álcool — ordenação natural (ex: nunca < às vezes < frequentemente).

- **transportation_type**  
  categórica (nominal)  
  Descreve o principal modo de transporte (ex: caminhada, transporte público, carro) — nominal, sem ordem.

## Coluna alvo (target)

- **obesity_level**  
  categórica (ordinal)  
  Mapeada para inteiros para refletir a gravidade crescente:  
  `Insufficient_Weight = -1`, `Normal_Weight = 0`, ..., `Obesity_Type_III = 5`.