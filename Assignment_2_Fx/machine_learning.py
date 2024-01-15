import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_evaluate_ano(df, features, target):
    df = df.dropna(subset=[target])
    df[features] = df[features].fillna(df[features].mean())
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/anonymized.png")

    return model, r2, mse

def train_and_evaluate_orignal(df, features, target):
    df = df.dropna(subset=[target])
    df[features] = df[features].fillna(df[features].mean())
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/machine_learning/original.png")

    return model, r2, mse



def train_and_evaluate_comparison(train_df, test_df, features, target, sample_size=1000):
    train_df = train_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    train_sample = train_df.sample(n=100000, random_state=42)
    test_sample = test_df.sample(n=20000, random_state=42)

    train_sample = train_sample.dropna(subset=features + [target])
    test_sample = test_sample.dropna(subset=features + [target])
    train_sample[features] = train_sample[features].fillna(train_sample[features].mean())
    test_sample[features] = test_sample[features].fillna(test_sample[features].mean())

    X_train = train_sample[features]
    y_train = train_sample[target]
    X_test = test_sample[features]
    y_test = test_sample[target]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.hexbin(y_test, y_pred, gridsize=50, cmap='Blues')
    plt.colorbar(label='Nombre dans le bin')
    plt.xlabel('Actual (Anonymized)')
    plt.ylabel('Predicted (Model on Original)')
    plt.title('Hexbin Actual vs Predicted (Anonymized)')

    plt.subplot(1, 2, 2)
    sns.histplot(y_test, color="blue", kde=True, label='Actual', stat="density", linewidth=0)
    sns.histplot(y_pred, color="red", kde=True, label='Predicted', stat="density", linewidth=0)
    plt.xlabel('deadlift')
    plt.ylabel('Density')
    plt.title('Distribution of Actual vs Predicted (Anonymized)')
    plt.legend()

    plt.tight_layout()

    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/machine_learning/selected_data.png")

    return model, r2, mse



def remove_outliers(data, feature_names):
    Q1 = data[feature_names].quantile(0.25)
    Q3 = data[feature_names].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data[feature_names] < (Q1 - 1.5 * IQR)) |(data[feature_names] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data




def machine_learning():
    original_data = 'Assignment_2_Fx/Assignment_2_Data_sets/selected_data.csv'
    anonymized_data = 'Assignment_2_Fx/Assignment_2_Data_sets/selected_data.csv'



    features = ['age', 'height', 'weight', 'candj', 'snatch', 'backsq']
    target = 'deadlift'

    original_data = pd.read_csv(original_data)
    anonymized_data = pd.read_csv(anonymized_data)

    original_data = remove_outliers(original_data, ['age', 'height', 'weight', 'candj', 'snatch', 'backsq', 'deadlift'])
    anonymized_data = remove_outliers(anonymized_data, ['age', 'height', 'weight', 'candj', 'snatch', 'backsq', 'deadlift'])

    model_original, r2_original, mse_original = train_and_evaluate_orignal(original_data, features, target)
    print("Original R² : ", r2_original, ", MSE : ", mse_original)
    model_anonymized, r2_anonymized, mse_anonymized = train_and_evaluate_ano(anonymized_data, features,target)

    print("Anonymized R² : ", r2_anonymized, ", MSE : ", mse_anonymized)

    model_combined, r2_combined, mse_combined = train_and_evaluate_comparison(anonymized_data, original_data, features, target)
    print("Comparison R² : ", r2_combined, ", MSE : ", mse_combined)

    results_df = pd.DataFrame({
    'R2_Anonymized': [r2_anonymized],
    'MSE_Anonymized': [mse_anonymized],
    'R2_Combined': [r2_combined],
    'MSE_Combined': [mse_combined]

    })

    results_csv_path = 'Assignment_2_Fx/Assignment_2_Data_sets/machine_learning/selected_data.csv'
    results_df.to_csv(results_csv_path, index=False)


