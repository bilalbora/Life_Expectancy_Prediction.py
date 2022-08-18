import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import helper as hp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


### Columns:
# Adult mortality = 1000 kişide 15 ile 60 yaş arasında ölme ihtimali
# Infant Deaths = 1000 kişi başına ölen bebek sayısı
# Percentage Expenditure = Milli hasılada sağlık harcama oranı
# Hepatits B = Hepatit b aşılanma yüzdesi
# Measles Kızamık Oranı
# Polio = 1 yaş bebek çocuk aşı kapsamı
# Total Expenditure = Hükümetin toplam sağlık harcama yüzdesi
# Diptheria = 1 yaş aşılama kapsamı
# GDP = Kişi başına düşen gayrisafi milli hasıla
# thinnes = Zayıflık aralığı
# Income composition = Kaynakların insanlar için kullanımı 0-1 arası
# Schooling = Eğitim yılı sayısı

df = pd.read_csv('Life Expectancy Data.csv')

hp.check_df(df)

df.rename(columns={" BMI ":"BMI","Life expectancy ":"Life_Exp", "Adult Mortality":"Adult_Mortality",
                   "infant deaths":"Infant_Deaths","percentage expenditure":"Perc_Exp","Hepatitis B":"HepB",
                  "Measles ":"Measles"," BMI ":"BMI","under-five deaths ":"Under_Five_Deaths","Diphtheria ":"Diphtheria",
                  " HIV/AIDS":"HIV/AIDS"," thinness  1-19 years":"thinness_1to19_years"," thinness 5-9 years":"thinness_5to9_years","Income composition of resources":"Income_Comp_Of_Resources",
                   "Total expenditure":"Tot_Exp"},inplace=True)

##########################           EDA             ############################

cat_cols, num_cols, cat_but_car = hp.grab_col_names(df)
cat_cols.append('Country')
cat_cols.append('Year')

for col in cat_cols:
    hp.cat_summary(df, col)

for col in num_cols:
    hp.num_summary(df, col)

for col in cat_cols:
    hp.target_summary_with_cat(df,'Life expectancy ', col)

for col in num_cols:
    hp.target_summary_with_num(df, 'Life expectancy ', col)

hp.high_correlated_cols(df)
df.drop('Under_Five_Deaths', axis=1, inplace=True)
num_cols.remove('Under_Five_Deaths')


############################            EKSİK VE AYKIRI DEĞER ANALİZİ       ##########################

hp.check_class(df)
hp.desc_stats(df)
df.dropna(inplace=True)

for col in num_cols:
    print(col, hp.check_outlier(df, col))

for col in num_cols:
    if hp.check_outlier(df, col):
        hp.replace_with_thresholds(df, col)

####################        FEATURE ENGINEERING     ###########################
dff = df.copy()
df = dff.copy()

df.loc[(df['Status'] == 'Developing') & (df['Population'] > df['Population'].mean()), 'New_Stat_Pop'] = 'Developing_Over_Pop'
df.loc[(df['Status'] == 'Developing') & (df['Population'] < df['Population'].mean()), 'New_Stat_Pop'] = 'Developing_Under_Pop'
df.loc[(df['Status'] == 'Developed') & (df['Population'] > df['Population'].mean()), 'New_Stat_Pop'] = 'Developed_Over_Pop'
df.loc[(df['Status'] == 'Developed') & (df['Population'] < df['Population'].mean()), 'New_Stat_Pop'] = 'Developed_Under_Pop'
df.groupby(by = 'New_Stat_Pop')['Life_Exp'].mean()

df['Toplam_Harcama_Katsayısı'] = df['Perc_Exp'] * df['Tot_Exp']

df['Gdp_Income'] = df['GDP'] / df['Income_Comp_Of_Resources']

df['New_POL_DİPT'] = df['Polio'] + df['Diphtheria']

df.head()


#####################       ENCODING        ##########################
cat_cols, num_cols, cat_but_car = hp.grab_col_names(df)
cat_cols.append('Country')
df = hp.one_hot_encoder(df, cat_cols, drop_first=True)

#####################           SCALING         ##################
num_cols.remove('Year')
num_cols = [col for col in num_cols if col not in ["Life_Exp"]]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

#########################           MODEL KURMA         ########################

y = df["Life_Exp"]
X = df.drop(["Life_Exp"], axis=1)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")



##############      HYPERMATER OPTIMIZATION     #############

gbm_model = GradientBoostingRegressor(random_state=17)

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse


#######################         FEATURE IMPORTANCE      ###################

hp.plot_importance(gbm_final, X, len(X))