import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load the dataset
data = pd.read_csv("Student Mental health.csv")

# One-hot encode categorical variables
X = pd.get_dummies(data.drop(columns=["Did you seek any specialist for a treatment?"]))

# Target variable
y = data["Did you seek any specialist for a treatment?"]

# Convert target variable to numerical (optional)
# You may not need to convert the target variable if it's already encoded properly
y = y.replace({"No": 0, "Yes": 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.astype('float32'), y.astype('float32'), test_size=0.2, random_state=42)

# Model architecture
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Model compilation
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Model evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Save the trained model
model.save("mental_health_diagnosis_model.h5")
