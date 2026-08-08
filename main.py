import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def main():
    # 1. Load the Boston Housing dataset
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)

    # 2. Data Cleaning: Drop NaNs, invalid rows, and duplicates
    df = df.dropna()
    df = df[df['medv'] > 0]
    df = df.drop_duplicates()

    # 3. Features (X) and Target Vector (y)
    # 'medv' = median home value in $1,000s
    X, y = df.drop('medv', axis=1), df['medv']

    # 4. Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("--- Model Performance ---")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Absolute Error: ${mae * 1000:,.2f}")

if __name__ == "__main__":
    main()