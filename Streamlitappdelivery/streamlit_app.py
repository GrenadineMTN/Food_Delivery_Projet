import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib
import base64


# Charger le modèle
pipeline = joblib.load('random_forest_model.joblib')

# Charger les données
df = pd.read_csv('df2.csv')

# Renommer la colonne
df.rename(columns={'Time_Difference': 'Time_preparation'}, inplace=True)

# Définir les valeurs par défaut pour le formulaire en fonction des valeurs minimales et maximales
default_values = {
    'Delivery_person_Age': df['Delivery_person_Age'].min(),
    'Delivery_person_Ratings': df['Delivery_person_Ratings'].min(),
    'Hour_picked': df['Hour_picked'].min(),
    'multiple_deliveries': df['multiple_deliveries'].min(),
    'Time_preparation': df['Time_preparation'].min(),
    'distance': df['distance'].min(),
    'Weatherconditions': df['Weatherconditions'].mode()[0],
    'Road_traffic_density': df['Road_traffic_density'].mode()[0],
    'Vehicle_condition': df['Vehicle_condition'].min(),
    'Type_of_order': df['Type_of_order'].mode()[0],
    'Type_of_vehicle': df['Type_of_vehicle'].mode()[0],
    'Festival': df['Festival'].mode()[0],
    'City': df['City'].mode()[0],
    'City_code': df['City_code'].mode()[0],
    'Time_category': df['Time_category'].mode()[0],
    'is_weekend': df['is_weekend'].min(),
    'day_of_week': df['day_of_week'].min()  # Ajouter la colonne day_of_week avec sa valeur par défaut
}

# Charger l'image du logo
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("livreur1.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url(data:image/jpeg;base64,{img});
background-size: contain;
background-position: center;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('Delivery Time Prediction')

# Créer un formulaire pour saisir les valeurs
st.sidebar.header('Enter Delivery Details')
user_input = {}
for feature, default_value in default_values.items():
    if feature != 'is_weekend':  # Exclure temporairement la colonne is_weekend du formulaire
        if feature in ['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City', 'City_code', 'Time_category']:
            user_input[feature] = st.sidebar.selectbox(feature, df[feature].unique(), index=list(df[feature].unique()).index(default_value))
        else:
            user_input[feature] = st.sidebar.number_input(feature, value=default_value, min_value=df[feature].min(), max_value=df[feature].max())

# Mettre is_weekend à 1 si le jour de la semaine est 5 (Samedi) ou 6 (Dimanche)
if 'day_of_week' in user_input:
    if user_input['day_of_week'] in [5, 6]:
        user_input['is_weekend'] = 1
    else:
        user_input['is_weekend'] = 0

# Ajouter la colonne is_weekend dans les valeurs par défaut
default_values['is_weekend'] = user_input.get('is_weekend', default_values['is_weekend'])

# Ajouter la colonne is_weekend dans le formulaire avec sa valeur par défaut
user_input['is_weekend'] = st.sidebar.selectbox('Is Weekend', [0, 1], index=default_values['is_weekend'])

# Créer une DataFrame à partir des valeurs saisies
user_df = pd.DataFrame([user_input])

# Prédire le temps de livraison en utilisant le modèle chargé
predicted_time = pipeline.predict(user_df)

# Afficher le temps de livraison prédit
st.subheader('Predicted Delivery Time')
st.write(f'The predicted delivery time is {predicted_time[0]:.2f} minutes.')
