import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model():
    
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_test = pd.read_csv('data/y_test.csv')
    
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train.values.ravel())
    
    
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    
    joblib.dump(model, 'models/landslide_model.pkl')
    print("\nModel saved successfully!")

if __name__ == "__main__":
    train_model()