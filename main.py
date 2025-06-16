# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import pandas as pd

# # Load models and encoders
# nb_diet = joblib.load('nb_diet_model.pkl')
# nb_exercise = joblib.load('nb_exercise_model.pkl')
# label_encoders = joblib.load('label_encoders.pkl')
# scaler = joblib.load('scaler.pkl')

# # Define the FastAPI app
# app = FastAPI()

# # Pydantic model for input validation
# class UserInput(BaseModel):
#     age: int
#     gender: str
#     weight: float
#     height: float
#     weight_status: str
#     health_condition: str
#     dietary_preference: str

# # Function to make predictions based on user input
# def make_predictions(age, gender, weight, height, weight_status, health_condition, dietary_preference):
#     # Encode and scale inputs
#     try:
#         gender = label_encoders['Gender'].transform([gender])[0]
#         weight_status = label_encoders['Weight Status'].transform([weight_status])[0]
        
#         # Handle 'None' case for health condition
#         if health_condition == 'None':
#             health_condition_encoded = -1  # Using -1 to indicate 'None' (not seen in training data)
#         else:
#             health_condition_encoded = label_encoders['Health Condition'].transform([health_condition])[0]
        
#         dietary_preference = label_encoders['Dietary Preference'].transform([dietary_preference])[0]
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")
    
#     input_data = pd.DataFrame([[age, weight, height]], columns=['Age', 'Weight', 'Height'])
#     scaled_features = scaler.transform(input_data)
#     age, weight, height = scaled_features[0]
    
#     X_input = pd.DataFrame([[age, gender, weight, height, weight_status, health_condition_encoded, dietary_preference]],
#                            columns=['Age', 'Gender', 'Weight', 'Height', 'Weight Status', 'Health Condition', 'Dietary Preference'])
    
#     # Make predictions
#     diet_pred = nb_diet.predict(X_input)[0]
#     exercise_pred = nb_exercise.predict(X_input)[0]
    
#     # Calculate probabilities
#     diet_probs = nb_diet.predict_proba(X_input)[0]
#     exercise_probs = nb_exercise.predict_proba(X_input)[0]
    
#     # Decode predictions
#     diet_pred = label_encoders['Diet'].inverse_transform([diet_pred])[0]
#     exercise_pred = label_encoders['Exercise'].inverse_transform([exercise_pred])[0]
    
#     # Find the probability of the predicted class
#     diet_pred_prob = max(diet_probs)  # Probability of the predicted diet class
#     exercise_pred_prob = max(exercise_probs)  # Probability of the predicted exercise class
    
#     return diet_pred, exercise_pred, diet_pred_prob, exercise_pred_prob

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.post("/predict")
# async def predict(user_input: UserInput):
#     try:
#         diet, exercise, diet_prob, exercise_prob = make_predictions(
#             user_input.age,
#             user_input.gender,
#             user_input.weight,
#             user_input.height,
#             user_input.weight_status,
#             user_input.health_condition,
#             user_input.dietary_preference
#         )
#         return {
#             "Recommended Diet": diet,
#             "Diet Prediction Confidence": diet_prob,
#             "Recommended Exercise": exercise,
#             "Exercise Prediction Confidence": exercise_prob
#         }
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# # Valid options for each field
# valid_genders = [str(x) for x in label_encoders['Gender'].classes_]
# valid_weight_statuses = [str(x) for x in label_encoders['Weight Status'].classes_]
# valid_health_conditions = [str(x) for x in label_encoders['Health Condition'].classes_]
# valid_health_conditions = ['None' if x == 'nan' else x for x in valid_health_conditions]
# valid_dietary_preferences = [str(x) for x in label_encoders['Dietary Preference'].classes_]

# @app.get("/options")
# async def get_options():
#     return {
#         "Valid Genders": valid_genders,
#         "Valid Weight Statuses": valid_weight_statuses,
#         "Valid Health Conditions": valid_health_conditions,
#         "Valid Dietary Preferences": valid_dietary_preferences
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Load models and encoders
nb_diet = joblib.load('nb_diet_model.pkl')
nb_exercise = joblib.load('nb_exercise_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Define the FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for input validation
class UserInput(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    weight_status: str
    health_condition: str
    dietary_preference: str

# Function to make predictions based on user input
def make_predictions(age, gender, weight, height, weight_status, health_condition, dietary_preference):
    # Encode and scale inputs
    try:
        gender = label_encoders['Gender'].transform([gender])[0]
        weight_status = label_encoders['Weight Status'].transform([weight_status])[0]
        
        # Handle 'None' case for health condition
        if health_condition == 'None':
            health_condition_encoded = -1  # Using -1 to indicate 'None' (not seen in training data)
        else:
            health_condition_encoded = label_encoders['Health Condition'].transform([health_condition])[0]
        
        dietary_preference = label_encoders['Dietary Preference'].transform([dietary_preference])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")
    
    input_data = pd.DataFrame([[age, weight, height]], columns=['Age', 'Weight', 'Height'])
    scaled_features = scaler.transform(input_data)
    age, weight, height = scaled_features[0]
    
    X_input = pd.DataFrame([[age, gender, weight, height, weight_status, health_condition_encoded, dietary_preference]],
                           columns=['Age', 'Gender', 'Weight', 'Height', 'Weight Status', 'Health Condition', 'Dietary Preference'])
    
    # Make predictions
    diet_pred = nb_diet.predict(X_input)[0]
    exercise_pred = nb_exercise.predict(X_input)[0]
    
    # Calculate probabilities
    diet_probs = nb_diet.predict_proba(X_input)[0]
    exercise_probs = nb_exercise.predict_proba(X_input)[0]
    
    # Decode predictions
    diet_pred = label_encoders['Diet'].inverse_transform([diet_pred])[0]
    exercise_pred = label_encoders['Exercise'].inverse_transform([exercise_pred])[0]
    
    # Find the probability of the predicted class
    diet_pred_prob = max(diet_probs)  # Probability of the predicted diet class
    exercise_pred_prob = max(exercise_probs)  # Probability of the predicted exercise class
    
    return diet_pred, exercise_pred, diet_pred_prob, exercise_pred_prob

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(user_input: UserInput):
    try:
        diet, exercise, diet_prob, exercise_prob = make_predictions(
            user_input.age,
            user_input.gender,
            user_input.weight,
            user_input.height,
            user_input.weight_status,
            user_input.health_condition,
            user_input.dietary_preference
        )
        return {
            "Recommended Diet": diet,
            "Diet Prediction Confidence": diet_prob,
            "Recommended Exercise": exercise,
            "Exercise Prediction Confidence": exercise_prob
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Valid options for each field
valid_genders = [str(x) for x in label_encoders['Gender'].classes_]
valid_weight_statuses = [str(x) for x in label_encoders['Weight Status'].classes_]
valid_health_conditions = [str(x) for x in label_encoders['Health Condition'].classes_]
valid_health_conditions = ['None' if x == 'nan' else x for x in valid_health_conditions]
valid_dietary_preferences = [str(x) for x in label_encoders['Dietary Preference'].classes_]

@app.get("/options")
async def get_options():
    return {
        "Valid Genders": valid_genders,
        "Valid Weight Statuses": valid_weight_statuses,
        "Valid Health Conditions": valid_health_conditions,
        "Valid Dietary Preferences": valid_dietary_preferences
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

