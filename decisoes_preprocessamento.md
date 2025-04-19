## feature columns

- gender  
    categorical (nominal)  
    Values: Male, Female  

- age  
    numerical (float)  
    Rounded to nearest integer to represent completed years — avoids unnecessary model sensitivity to decimals (e.g., 20.7 vs 21.3).  
    Converted to float to maintain consistency in dtype.

- height  
    numerical (float)  
    Rounded to 2 decimal places to reduce noise and avoid excessive granularity.

- weight  
    numerical (float)  
    Rounded to 2 decimal places for the same reason as height — cleaner input for the model.

- family_history_overweight  
    categorical (nominal)  
    Binary yes/no feature — converted to categorical dtype.

- high_caloric_food  
    categorical (nominal)  
    Indicates whether the person frequently consumes high-caloric food — treated as a nominal category.

- freqof_vegetables  
    categorical (ordinal)  
    Frequency of vegetable consumption — has a natural order (e.g., never < sometimes < always).

- number_of_main_meals  
    numerical (discrete, ratio-type)  
    Number of main meals per day — treated as numeric since the range is relatively wide and meaningful numerically.  
    Rounded and stored as float.

- food_between_meals  
    categorical (ordinal)  
    Describes snacking frequency — has a natural order (e.g., never < sometimes < frequently).

- smoking  
    categorical (nominal)  
    Binary yes/no variable — treated as nominal category.

- water_intake  
    categorical (ordinal)  
    Frequency of daily water consumption — values have a clear progression.

- calories_monitoring  
    categorical (nominal)  
    Binary feature indicating whether the person monitors their caloric intake — treated as nominal.

- freq_physical_activity  
    categorical (ordinal)  
    Ordered levels of physical activity frequency — higher value = more activity.

- time_using_technology  
    categorical (ordinal)  
    Time spent on electronic devices — more time implies more sedentary behavior, hence treated as ordinal.

- alcohol_consumption  
    categorical (ordinal)  
    Frequency of alcohol consumption — naturally ordered (e.g., never < sometimes < frequently).

- transportation_type  
    categorical (nominal)  
    Describes the main mode of transport (e.g., walking, public transport, car) — nominal without order.

## target column

- obesity_level  
    categorical (ordinal)  
    Mapped to integers to reflect increasing severity:  
    `Insufficient_Weight = -1`, `Normal_Weight = 0`, ..., `Obesity_Type_III = 5`.