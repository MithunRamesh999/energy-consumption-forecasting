from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

def train_models(df):

    if df.shape[0] < 10:
        raise ValueError(
            f"Not enough data to train models. Rows available: {df.shape[0]}"
        )

    X = df.drop(['Timestamp', 'Energy_Consumption'], axis=1)
    y = df['Energy_Consumption']

    # Handle NaN values (CRITICAL FIX)
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]



    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, X_test, y_test, imputer
