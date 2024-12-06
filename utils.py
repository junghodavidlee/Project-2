from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Trains a given model, makes predictions, and evaluates performance.

    Parameters:
    model (object): A machine learning model with `fit` and `predict` methods.
    X_train (array-like): Training features.
    y_train (array-like): Training labels.
    X_test (array-like): Test features.
    y_test (array-like): Test labels.
    
    """
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))