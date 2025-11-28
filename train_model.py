import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json

df = pd.read_csv('WineQT.csv')

X = df.drop(['quality', 'Id'], axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred) * 100
nb_f1 = f1_score(y_test, nb_pred, average='weighted')


if nb_f1:
    model = nb_model
    model_name = "Naive Bayes"

joblib.dump(model, 'api/model.pkl')
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'api/feature_names.pkl')

performance_data = {
    "naive_bayes": {
        "accuracy": round(nb_accuracy, 4),
        "f1_score": round(nb_f1, 4)
    },
    "model_name": model_name
}

with open('api/model_performance.json', 'w') as f:
    json.dump(performance_data, f, indent=4)

print("\nModel terbaik, nama fitur, dan data kinerja telah disimpan.")
