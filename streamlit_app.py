import streamlit as st
import pandas as pd 
import numpy as np
from PIL import Image
import pickle
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import folium
import math

from streamlit_folium import st_folium

from shapely.geometry import  Point




# DF light pour affichage run rapide
lfb = pd.read_csv("Data csv/df.csv")
image = Image.open("st_image/deco/Title_header.png")
st.image(image, use_column_width=True) 


st.sidebar.title("Sommaire")
pages = ['Projet','Analyse Exploiratoire des donnés',"Data Visualisation",'Statistiques', 'Analyse Modèle', "Classification","Regression Map", 'Conclusion']
page = st.sidebar.radio("Aller vers", pages)

#------------------------------------------------------ Projet -------------------------------------------------------------------------
if page == pages[0]:
    
    intro = Image.open("st_image/deco/intro.png")
    st.image(intro)
    
    word = Image.open("st_image/deco/WordCloud.png")
    st.image(word)
#-----------------------------------------------------Analyse de Données & Preproceesing
if page == pages[1]:
    
    st.title("Preprocessing")
    st.dataframe(lfb.head(10))
    st.write(lfb.shape)
    #st.dataframe(lfb.describe())
    col1, col2 = st.columns(2)
    with col1:
     if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(lfb.isna().sum())
    with col2:
        if st.checkbox("Afficher les doublons") : 
            st.write(lfb.duplicated().sum())
    st.title("Feature engineering")
    # Création d'un DataFrame pour stocker les labellisations
    labels_df = pd.DataFrame({
    'Variable': ['Label_journee', 'day_name', 'Duree_inter_label', 'total_label', 'Label_mean_mob', 'Label_response','label_pop'],
    'Correspondance': ['Catégorie par moment de la journée', 'Jour de la semaine', 'Durée de l\'intervention', 'Total de mobilisation', 'Par moyenne de station', 'Label de la variable response time','Population par Borough'],
    'Labels': [
        lfb['Label_journee'].unique(), 
        lfb['day_name'].unique(), 
        lfb['Duree_inter_label'].unique(), 
        lfb['total_label'].unique(), 
        lfb['Label_mean_mob'].unique(), 
        lfb['Label_response'].unique(),
        lfb['label_pop'].unique(),
    ]
    })

    # Affichage du DataFrame dans Streamlit
    st.subheader("Labellisations des Variables")
    st.write(labels_df)

    # Création de la colonne de prix médian des maisons
    house = pd.read_excel("Data csv/house_london.xlsx").rename(columns={"Unnamed: 0": 'name'}).drop([33,34], axis=0)

    price_house = {n:i for n,i in zip(house['name'], house["Median house price"])}
    lfb['price_house_median'] = lfb['Borough'].map(price_house)
    lfb["label_price"] = pd.qcut(lfb['price_house_median'], q=[0, 0.25, 0.75, 1], labels=['low_cost', 'median', 'high_cost'])

    # Application Streamlit
    def main():
        st.title("Analyse des prix médians des maisons par Borough")


    # Affichage des données
    st.write("Voici un échantillon des données avec la nouvelle fonctionnalité de prix médian des maisons :")
    st.write(lfb[['Borough','label_pop', 'price_house_median', 'label_price','Label_response',]].head(10))
  
# ----------------------------------------------------Data Visualisation ------------------------------------------------------------    
if page == pages[2]:
    bi1 = Image.open("st_image/BI12.png")
    st.image(bi1, use_column_width=True)
    bi2 = Image.open('st_image/BI2.png')
    st.image(bi2, use_column_width=True)
#------------------------------------------------Statistique -----------------------------------------------------------
if page == pages[3]:
    st.title("Analyse statistique")
    
    st.subheader("Effet de la densité de la population sur le temps d'intervention")
    st.subheader("Test de Levene  ")
    # Sélectionner les échantillons pour chaque niveau de densité de population
    g1 = lfb.loc[lfb['label_pop'] == "Faible densité"]['Duree_intervention']
    g2 = lfb.loc[lfb['label_pop'] == "Dense"]['Duree_intervention']
    g3 = lfb.loc[lfb['label_pop'] == "Très dense"]['Duree_intervention']

    
    
    result_df = pd.DataFrame({
    'Group': ['Faible densité', 'Dense', 'Très dense'],
    'Test Statistic': [4.859388333710579, 4.859388333710579, 4.859388333710579],
    'p-value': [0.007764386325863089, 0.007764386325863089, 0.007764386325863089]
    })

    st.write(result_df)
    
    st.write("Statistique de Levene :", 4.859388333710579)
    st.write("Valeur p :", 0.007764386325863089)
    
    p = 0.007764386325863089
    if p < 0.05:
        st.write("Interpretation: p < 0.05,Il y a donc une différence significative dans la variance de la durée d'intervention entre les groupes.")
        
    else:
        st.write("Il n'y a pas de différence significative dans la variance de la durée d'intervention entre les groupes.")
    
    # Test anonva
    st.subheader('Anova robust k=3')
    
    
    st.write('Resultat test Anova de type robuste pour compenser la violation du test Levene.')
    res = pd.read_csv('Data csv/anova.csv')
  
    st.write(res)    
    #Test de Turkey 
    st.subheader("Test de Tukey")
    
    st.subheader(" Multiple Comparison of Means - Tukey HSD, FWER=0.05")
    turkey = pd.read_csv("Data csv/Turkey.csv")
    st.write(turkey)
    st.write("La différence entre la moyenne de faible densité et très dense est la plus importante (131.902) ce qui indique la plus grande magnitude.\n \
            On peut affirmer avec moins de 5% de se tromper que la durée d'intervention est plus longue dans les zones très dense.")
    
    st.subheader("Analyse du temps d'intervention en fonction du prix de l'immobilier")

    st.subheader("Test Annova robuste")
    anova_robuste_result = pd.read_csv('Data csv/anova_label.csv')

    st.write(anova_robuste_result)
    st.write("La valeur de p associée à ce test est de 0, ce qui indique une différence significative entre les groupes.\n \
            Cependant, il est important de noter que l'effect size (η²) est très faible, avec une valeur de 0.0116")
    
    st.subheader("Test de Tukey")
    
    tukey_results_df = pd.read_csv("Data csv/tukey_label.csv")
    st.subheader(" Multiple Comparison of Means - Tukey HSD, FWER=0.05   ")
    st.write(tukey_results_df)
    st.write("Le meandiff avec le plus de magnitude concerne les zones dont le prix de l'immobilier est élevé contre les zones où les prix son bas. \nOn peut affirmer avec moins de 5{%} de chance de se tromper que le temps d'intervention est plus rapide pour les zones dont le prix de l'immobilié est élevé.")
    
    st.subheader("Analyse du temps d'intervention en fonction du prix de l'immobilier et population")
    st.subheader("Test ANOVA à 2 facteurs")
    
    aov = pd.read_csv("Data csv/inter_anova.csv")
    st.write(aov)  
    st.write("Toutes les valeurs p sont très proches de zéro, ce qui signifie que l'effet des différents niveaux de facteur, ainsi que leur interaction, sont statistiquement significatifs. \n Cela confirme notre hypothése qu'il existe une interaction significative entre le prix de l'immobilier, la durée d'intervention et la population sur la variable de réponse.")                   
#----------------------------------------------Modélisation------------------------------------------------------------------
if page == pages[4]:
    tab1, tab2 = st.tabs(['Regression', "Classification"])
    
    with tab1:
        
        st.header("Evaluation du modèle DecisionTreeRegressor")
             
        
        rmse_df = pd.read_csv("Data csv/rmse_tab_streamlit.csv")
        st.write(rmse_df)
        
        important = Image.open('st_image/courbe/feature.png')
        st.title("Graphique de regression")
        st.image(important, use_column_width=True)    
        
        courbe_tree = Image.open('st_image/courbe/linear.png')
        st.title("Graphique de regression")
        st.image(courbe_tree, use_column_width=True)
        
        courbe_tree = Image.open('st_image/courbe/residu.png')
        st.title("Graphique de résidu")
        st.image(courbe_tree, use_column_width=True)
        
        courbe_learning = Image.open("st_image/courbe/apprentissage.png")
        st.title("Courbe d'apprentissage")
        st.image(courbe_learning, use_column_width=True)
        
        courbe_error= Image.open("st_image/courbe/error.png")
        st.title("Courbe Apprentissage erreur")
        st.image(courbe_error, use_column_width=True)
        
        with tab2:
            st.header("Evaluation du modèle DecisionTreeClassification")
            matrice = Image.open('st_image/courbe/Confusion.png')
            st.title("Matrice de Confusion")
            st.image(matrice, use_column_width=True)
            
            st.title("Rapport de Classification")
            report = pd.read_csv('Data csv/report.csv')
            st.write(report)
            
#--------------------------------------------Classification ---------------------------------------------------------------     
if page == pages[5]:
    
    st.title('Prédiction Classifier')
    @st.cache_data

    def charger_modele(model_name):
    # Charger le modèle à partir du fichier Pickle
        with open(model_name, 'rb') as fichier_modele:
            modele = pickle.load(fichier_modele)
            return modele
    
    st.write('Classification')
    

    modele = charger_modele('Classification_Stream.pkl')
    
    #lon = st.slider("longitude", -0.50999133, 0.31974323, 0.0000001, format="%.6f")
    #lat = st.slider("latitude", 51.29251519, 51.69166991, 0.00000001, format="%.6f")
    casernes_lfb = {
    (-0.1290, 51.4975): "Westminster",
    (-0.1953, 51.4984): "Kensington",
    (-0.0711, 51.5246): "Shoreditch",
    (-0.1276, 51.5074): "City Center"
}

    
    
    lon_liste = [-0.068488, -0.148894, 0.027774, -0.124414]
    lat_liste = [51.633342, 51.475812, 51.322402, 51.476502	]
    Longitude_liste = [-0.1290, -0.1953, -0.0711, -0.1276]
    Latitude_liste = [51.4975, 51.4984, 51.5246, 51.5074]
    
    
    lon = st.selectbox("Choisissez une longitude", lon_liste)
    lat = st.selectbox("Choisissez une latitude", lat_liste)
    Longitude = st.selectbox("Choisissez une Longitude", Longitude_liste)
    Latitude=st.selectbox("Choisissez une Latitude", Latitude_liste)
    station_name = casernes_lfb.get((Longitude, Latitude), "Aucune caserne trouvée")
    st.write(f"Nom de la caserne : {station_name}")
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Convertir les latitudes et longitudes de degrés à radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        
        # Rayon moyen de la Terre en kilomètres
        R = 6371.0
        
        # Différences de latitude et de longitude
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Calcul de la distance haversine
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        return distance
    Distance = haversine_distance(lat, lon, Latitude, Longitude)
    
    pop_liste = [312500.0, 307000.0, 309400.0, 303100.0	]
    population = st.selectbox("Choisissez une population", pop_liste)
    
    
    prediction_reg = modele.predict([[Longitude,Latitude, lon, lat, population, Distance]])
    
    if st.button("Afficher la prédiction"):
        if prediction_reg == 0:
            st.write("Mobilisation Rapide")
        if prediction_reg == 1:
            st.write('Mobilisation Lente')
        if prediction_reg == 2:
            st.write("Mobilisation dans la Moyenne")

# ---------------------------------------------Map Regression ------------------------------------------------------------------------       
if page == pages[6]:
    st.title('Prédiction Regression with map')
    
    coord = pd.read_csv("Data csv/coordonee_ground.csv")
    population, Distance = 0,0
 # -------------------------------------------------------------- Dico Station -----------------------------------------------------------------------#   
    stations_dict = {}

# Parcourir les lignes du DataFrame pour créer le dictionnaire
    for index, row in coord.iterrows():
        station_name = row["Station"]
        latitude = row["Latitude"]
        longitude = row["Longitude"]
        stations_dict[station_name] = {"latitude": latitude, "longitude": longitude}
        
    # Créer une liste déroulante avec les noms des stations
    selected_station = st.selectbox("Sélectionnez une station", list(stations_dict.keys()))

    # Fonction pour obtenir les coordonnées de la station sélectionnée
    def get_coordinates(station_name):
        return stations_dict[station_name]["latitude"], stations_dict[station_name]["longitude"]

    # Afficher les coordonnées de la station sélectionnée
    latitude, longitude = get_coordinates(selected_station)
    st.write(f"Latitude: {latitude}, Longitude: {longitude}")
    
    
#------------------------------------------------------JS back coord --------------------------------------------------------------------------------#    

    def get_pos(lat, lng):
        return lat, lng

    # Afficher la carte une première fois
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
    folium.Marker(location=[latitude, longitude]).add_to(m)
    m.add_child(folium.LatLngPopup())
    
    mapa = st_folium(m, height=350, width=700)
    
    # Drapeau pour indiquer si l'utilisateur a cliqué sur la carte
    clicked = False

    # Si l'utilisateur clique sur la carte, mettre à jour le drapeau
    if mapa.get("last_clicked"):
        clicked = True

    # Définir lon et lat à None initialement
    lon, lat = None, None

    # Afficher la boucle seulement si l'utilisateur a cliqué sur la carte
    if clicked:
        data = None
        while data is None:
            # Obtenir les données une fois que l'utilisateur clique sur la carte
            if mapa.get("last_clicked"):
                data = get_pos(mapa["last_clicked"]["lat"], mapa["last_clicked"]["lng"])

        # Utiliser les données récupérées pour votre modèle
        if data is not None and len(data) >= 2:  # Assurez-vous que data contient au moins 2 éléments
            st.write("Coordonnées sélectionnées:", data) # Affiche les données récupérées
            lon, lat = data[1], data[0]

    # Utiliser lon et lat à l'extérieur de la boucle
    if lon is not None and lat is not None:
        st.write("")
        
# ----------------------------------------------------------------------Distance ---------------------------------------------------------------------------------

    def haversine_distance(lat1, lon1, lat2, lon2):
        # Convertir les latitudes et longitudes de degrés à radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        
        # Rayon moyen de la Terre en kilomètres
        R = 6371.0
        
        # Différences de latitude et de longitude
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Calcul de la distance haversine
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        return distance
    
# -----------------------------------------------------------------------------pop -----------------------------------------------------------------------------

    pop = pd.read_csv("Data csv/pop_borough.csv")
    pop_b = {b : p for b,p in zip(pop.Borough, pop.population)}
    
    import geopandas as gpd

    if lon is not None and lat is not None:
        f = gpd.read_file("london-topojson.json")
        point = Point(lon, lat)
        est_contenu = f.geometry.contains(point)

        resultats = f[est_contenu]
        population = pop_b[resultats['id'].iloc[0]]
    

# -----------------------------------------------------------------------------ML ------------------------------------------------------------------------------   
    @st.cache_data

    def charger_modele(model_name):
    # Charger le modèle à partir du fichier Pickle
        with open(model_name, 'rb') as fichier_modele:
            modele = pickle.load(fichier_modele)
        return modele

    modele= charger_modele('Streamlit_DecisionReg.pkl')
    
    Longitude = longitude
    Latitude = latitude
    
    if lon is not None and lat is not None:
        Distance = haversine_distance(Longitude, Latitude, lon, lat)
    
    
    prediction_reg = modele.predict([[Longitude,Latitude, lon, lat, population, Distance]])
    if st.button("Afficher la prédiction"):
     st.write(f"Temps de mobilisation: {np.round(prediction_reg/60,2)} min")

#-------------------------------------------------------------Conclusion-------------------------------------------------------------    
if  page == pages[7]:
    text = "L'ensemble de nos données ont pu mettre en évidence certain effet socio-démographique liée à la densité et au prix de l'immobilier en fonction des boroughs sur les délais de mobilisation. En effet, nous avons pu montrer que les arrondissements avec une faible densité ont un délai de mobilisation plus faible que les zones à forte densité. De la manière le prix de l'immobilier joue un rôle dans la modulation du délai de mobilisation. Plus les zones géographiques où le prix de l'immobilier est élevé plus le délai d'intervention diminue de la moyenne. Ces résultats peuvent être critiquable car ils ne permettent pas d'expliquer la cause de l'effet retrouvé. Plusieurs facteurs peuvent entrer en considération pour expliquer la variance comme le traffic routier, les politiques en œuvre pour favoriser l'accès au secours et bien d'autres exemple. En d’autres termes, ces résultats ne montrent en aucune manière le facteur humains seulement un lien entre des spécificité sociogéographique et le temps de réponse."
    st.write(text, text_align= 'justify')    
