# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import joblib
import tensorflow as tf
import os
import json
import pandas as pd
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
data = np.load("data/lstm_dataset.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# Load label encoder to get class names
le_dict = joblib.load("encoders.pkl")
le_attack_chain = le_dict["attack_chain"]
class_labels = le_attack_chain.classes_

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Build LSTM model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with early stopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# Save the trained model
model.save("model/lstm_attack_chain_model.keras")
print("Model saved as model/lstm_attack_chain_model.keras")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("output_training_history.png")
plt.show()

# Predict on test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("output_confusion_matrix.png")
plt.show()

# Multiclass ROC/AUC
fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_cat[:, i], y_pred_prob[:, i])
    roc_auc[i] = roc_auc_score(y_test_cat[:, i], y_pred_prob[:, i])

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("output_roc_auc.png")
plt.show()

# Print classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=class_labels))

# Feature importance note
print("\n=== Feature Importance Analysis ===")
feature_names = ['agent_ip', 'agent_name', 'agent_id', 'eventdata_image', 'rule_id', 'mitre_id']
print(f"Features used in model: {feature_names}")
print("Note: For detailed feature importance, consider using SHAP or permutation importance externally.")

# Save training metrics
output_metrics = {
    "accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"],
    "loss": history.history["loss"],
    "val_loss": history.history["val_loss"]
}
with open("output_training_metrics.json", "w") as f:
    json.dump(output_metrics, f, indent=4)

# Save confusion matrix CSV
np.savetxt("output_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

# Save classification report
report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
pd.DataFrame(report).transpose().to_csv("classification_report.csv")

# Save model info
model_info = f"""Model: LSTM Attack Chain Classifier
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input Features: image, ruleName, attack_stage
Model Path: model/lstm_attack_chain_model.keras
Label Classes: {', '.join(class_labels)}
"""
with open("model_info.txt", "w") as f:
    f.write(model_info)

# Save test results for dashboard
np.savez("data/lstm_dashboard_data.npz",
         y_test=y_test,
         y_pred=y_pred,
         y_test_cat=y_test_cat,
         y_pred_prob=y_pred_prob)

# Re-save label encoder if needed
if not os.path.exists("encoders.pkl"):
    joblib.dump({"attack_chain": le_attack_chain}, "encoders.pkl")
