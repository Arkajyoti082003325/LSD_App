import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data():
  
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    
    df = pd.read_csv('landslide_data.csv')  # Make sure the file is in your project directory
    
    
    print("Original dataset shape:", df.shape)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    
    if 'fatality_count' in df.columns:
        df['fatality_count'].fillna(0, inplace=True)
    if 'injury_count' in df.columns:
        df['injury_count'].fillna(0, inplace=True)
    if 'landslide_size' in df.columns:
        df['landslide_size'].fillna('unknown', inplace=True)
    if 'landslide_trigger' in df.columns:
        df['landslide_trigger'].fillna('unknown', inplace=True)
    
    
    if 'fatality_count' in df.columns and 'injury_count' in df.columns:
        df['impact_severity'] = np.where(
            (df['fatality_count'] > 0) | (df['injury_count'] > 0),
            'high',
            'low'
        )
    else:
        
        df['impact_severity'] = 'low'
    
    
    label_encoders = {}
    categorical_cols = []
    
    
    for col in ['landslide_size', 'landslide_trigger', 'impact_severity']:
        if col in df.columns:
            categorical_cols.append(col)
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        joblib.dump(le, f'models/{col}_label_encoder.pkl')
    
    
    features = []
    for col in ['latitude', 'longitude', 'landslide_size', 'landslide_trigger']:
        if col in df.columns:
            features.append(col)
    
    if not features:
        raise ValueError("No valid features found in the dataset")
    
    target = 'impact_severity' if 'impact_severity' in df.columns else None
    
    if not target:
        raise ValueError("Target column not found in the dataset")
    
    X = df[features]
    y = df[target]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print("\nPreprocessing completed!")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

if __name__ == "__main__":
    preprocess_data()