import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
import pickle

# Assuming you have trained the model and fitted the vectorizer in the previous cells, save them:
# Save the model
model.save("nutri_model.h5")  

# Save the vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Now load the trained model and vectorizer
model = keras.models.load_model("nutri_model.h5")
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
df = pd.read_csv("/content/drive/MyDrive/content/NutriAI1.csv", encoding='latin-1')

# FastAPI setup
app = FastAPI()

@app.get("/")
def home():
    return {"message": "NutriAI API is running!"}

@app.post("/recommend/")
def recommend(user_input: str, top_n: int = 5):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, vectorizer.transform(df["combined_text"])).flatten()
    related_indices = similarities.argsort()[-top_n:][::-1]
    results = df.iloc[related_indices][["Recipes", "Ingredients", "Calories", "Carbohydrates", "Steps"]].to_dict(orient="records")
    return {"recommendations": results}