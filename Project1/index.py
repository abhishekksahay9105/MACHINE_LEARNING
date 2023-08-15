#!/c/Users/z004kv5j/AppData/Local/Microsoft/WindowsApps/python3
import matplotlib
import scipy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


def main():
    housing = pd.read_csv ('./housing.csv')
    
    train_set, test_set = train_test_split (housing, test_size=0.2, random_state=42)
    

    split = StratifiedShuffleSplit (n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['CHAS']):
        start_train_set = housing.iloc[train_index]
        start_test_set = housing.iloc[test_index]

    housing = start_train_set.drop("MEDV", axis=1)
    housing_label = start_train_set["MEDV"].copy()

    my_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                        ('std_scaler', StandardScaler()),])
    
    housing_num_tr = my_pipeline.fit_transform(housing)

    model = LinearRegression()
    model.fit (housing_num_tr, housing_label)

    print (housing_num_tr)



if __name__ == "__main__":
    main()