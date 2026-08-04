# Preprocessing Report

## Dataset Summary
- Original rows: 145460
- Cleaned rows: 142193
- Training rows: 99535
- Validation rows: 21329
- Test rows: 21329

## Missing Value Summary
### Before
       Column  Missing_Count  Missing_Percentage Data_Type
     Sunshine          69835               48.01   float64
  Evaporation          62790               43.17   float64
     Cloud3pm          59358               40.81   float64
     Cloud9am          55888               38.42   float64
  Pressure9am          15065               10.36   float64
  Pressure3pm          15028               10.33   float64
   WindDir9am          10566                7.26    object
  WindGustDir          10326                7.10    object
WindGustSpeed          10263                7.06   float64
  Humidity3pm           4507                3.10   float64

### After
            Column  Missing_Count  Missing_Percentage Data_Type
          Sunshine          67816               47.69   float64
       Evaporation          60843               42.79   float64
          Cloud3pm          57094               40.15   float64
          Cloud9am          53657               37.74   float64
PressureDifference          14204                9.99   float64
       Pressure9am          14014                9.86   float64
       Pressure3pm          13981                9.83   float64
        WindDir9am          10013                7.04    object
       WindGustDir           9330                6.56    object
     WindGustSpeed           9270                6.52   float64

## Before vs After Statistics
             mean_before  std_before  min_before  max_before  mean_after  std_after  min_after  max_after
MinTemp        12.194034    6.398495        -8.5        33.9   12.190921   6.344066       -1.9       25.8
MaxTemp        23.221348    7.119049        -4.8        48.1   23.247765   6.975421        9.0       40.1
Rainfall        2.360918    8.478060         0.0       371.0    2.082348   5.864981        0.0       37.4
Humidity3pm    51.539116   20.795902         0.0       100.0   51.501245  20.719093        9.0       98.0

## Feature Distribution
- Feature distribution plot: reports\plots\feature_distributions.png

## Class Balance
- Class balance plot: reports\plots\target_balance.png