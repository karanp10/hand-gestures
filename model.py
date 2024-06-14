import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


def load_gesture_data(file_path):
    data = pd.read_csv(file_path)
    return data

def prepare_data(data):
    # Encode the handedness column
    le = LabelEncoder()
    data['handedness'] = le.fit_transform(data['handedness'])

    X = data.drop(columns=['gesture_name'])
    y = data['gesture_name']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy * 100:.2f}%')

    # Save the trained model
    joblib.dump(model, 'gesture_recognition_model.pkl')

    return model

if __name__ == '__main__':
    data = load_gesture_data('right_hand_fist.csv')
    X, y = prepare_data(data)

    model = train_model(X, y)
