import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load and prepare the dataset
df = pd.read_csv('diabetes.csv')

# Drop irrelevant columns as you did previously
df.drop(columns=['BloodPressure', 'SkinThickness', 'Insulin'], inplace=True)

X = df.drop(columns='Outcome')
Y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)

# Define the prediction function
def predict_diabetes(Pregnancies, Glucose, BMI, DiabetesPedigreeFunction, Age):
    # Create a feature vector from the input values
    input_data = [[Pregnancies, Glucose, BMI, DiabetesPedigreeFunction, Age]]
    
    # Make a prediction using the trained model
    prediction = model.predict(input_data)[0]
    
    # Return prediction (0 = No Diabetes, 1 = Diabetes)
    return "Diabetes" if prediction == 1 else "No Diabetes"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_diabetes,  # Function that makes predictions
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],  # Input fields for the user to enter data
    outputs="text",  # Output text showing whether the prediction is "Diabetes" or "No Diabetes"
    live=True  # Optional: Update the prediction live as the user inputs data
)

# Launch the interface
interface.launch()
