
# Online Python - IDE, Editor, Compiler, Interpreter

pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. بارگذاری داده‌ها
data = pd.read_csv("cicids2017_sample.csv")  # فایل نمونه از CICIDS 2017
data = data.dropna()  # حذف مقادیر خالی

# 2. انتخاب ویژگی‌ها و برچسب‌ها
features = ['Destination Port', 'Protocol', 'Flow Duration', 'Total Fwd Packets', 
            'Total Length of Fwd Packets', 'Flow IAT Mean', 'Fwd IAT Std', 'Bwd IAT Std']
target = 'Label'

X = data[features]
y = data[target]

# 3. کدگذاری برچسب‌ها
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # تبدیل حمله یا عادی به 0 و 1

# 4. نرمال‌سازی داده‌ها
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5. ساخت توالی زمانی
def create_sequences(X, y, seq_length=100):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y)

# 6. تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ساخت مدل LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())


# آموزش مدل
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ذخیره مدل
model.save("network_intrusion_model.h5")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# پیش‌بینی داده‌های تست
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# محاسبه معیارها
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# محاسبه Recall
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall}")
