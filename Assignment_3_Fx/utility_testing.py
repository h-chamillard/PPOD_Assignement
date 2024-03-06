import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



def utility_testing():

    original_data_path = 'Last_Assignment_Dataset/adults.csv'
    anonymized_data_path = 'new_adult_anonymized.csv'
    original_data = pd.read_csv(original_data_path)
    anonymized_data = pd.read_csv(anonymized_data_path)

    original_info = original_data.describe()
    anonymized_info = anonymized_data.describe()

    print(original_info, anonymized_info)

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    sns.histplot(original_data['age'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Age in Original Dataset')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(anonymized_data['age'], bins=20, kde=True, color='orange')
    plt.title('Distribution of Age in Anonymized Dataset')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    common_columns = set(original_data.columns).intersection(set(anonymized_data.columns))
    common_columns = list(common_columns)

    original_corr = original_data[common_columns].apply(lambda x: pd.factorize(x)[0]).corr()
    anonymized_corr = anonymized_data[common_columns].apply(lambda x: pd.factorize(x)[0]).corr()

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    sns.heatmap(original_corr, ax=ax[0], annot=True, fmt=".2f", cmap="coolwarm")
    ax[0].set_title('Correlation Matrix of Original Dataset')

    sns.heatmap(anonymized_corr, ax=ax[1], annot=True, fmt=".2f", cmap="coolwarm")
    ax[1].set_title('Correlation Matrix of Anonymized Dataset')

    plt.show()

    original_corr_values = original_corr.unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates()
    anonymized_corr_values = anonymized_corr.unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates()


    def prepare_data(df, target_variable):
        y = df[target_variable]
        X = df.drop(target_variable, axis=1)
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(exclude=['object']).columns
        numerical_transformer = 'passthrough'
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        return pipeline, X, y

    original_pipeline, X_original, y_original = prepare_data(original_data, 'age')
    anonymized_pipeline, X_anonymized, y_anonymized = prepare_data(anonymized_data, 'age')

    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_original, y_original, test_size=0.2,
                                                                            random_state=0)
    X_train_anon, X_test_anon, y_train_anon, y_test_anon = train_test_split(X_anonymized, y_anonymized, test_size=0.2,
                                                                            random_state=0)
    original_pipeline.fit(X_train_orig, y_train_orig)
    orig_predictions = original_pipeline.predict(X_test_orig)
    orig_mse = mean_squared_error(y_test_orig, orig_predictions)
    orig_r2 = r2_score(y_test_orig, orig_predictions)

    anonymized_pipeline.fit(X_train_anon, y_train_anon)
    anon_predictions = anonymized_pipeline.predict(X_test_anon)
    anon_mse = mean_squared_error(y_test_anon, anon_predictions)
    anon_r2 = r2_score(y_test_anon, anon_predictions)

    results_comparison = {
        'Original Data': {'MSE': orig_mse, 'R^2': orig_r2},
        'Anonymized Data': {'MSE': anon_mse, 'R^2': anon_r2},
    }

    print(results_comparison)
