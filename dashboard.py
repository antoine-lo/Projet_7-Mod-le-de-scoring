 #APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
 #100762
import streamlit as st
import numpy as np
import pandas as pd
import time
from Utils.toolbox import model_local_interpretation,lime_explain,fonction_comparaison,df_chain_explain,graphes_streamlit,transform_numerical_to_categorical
from Utils.model import PipelinePredictor
from Utils.conf import path_lime_explainer,path_pipeline_obj,path_test,path_train,threshold,col_for_explaination
from request_api import api_result

### Data path
path_df = path_test
path_train = path_train
model = PipelinePredictor(path_pipeline_obj)
path_explainer = path_lime_explainer

@st.cache #mise en cache de la fonction pour exécution unique
def chargement_data(path):
    df = pd.read_csv(path)
    df = df.loc[:,~df.columns.str.startswith('Unnamed')]
    df = transform_numerical_to_categorical(df)
    return df

def main(path_test, path_train):

    df = chargement_data(path_test)
    df_train = chargement_data(path_train)
    liste_id = df['SK_ID_CURR'].tolist()

    #affichage formulaire
    st.markdown('# Outil d\'aide à la décision pour les prêts bancaires #\n')
    st.markdown("### C'est un outils d'analyse de risque lié à l'accord ou non d'un prêt bancaire. Un moteur renvoit pour les différents clients sélectionnés un score indiquant la capacité du client à remboursé son prêt ###")

    st.sidebar.title('Veuillez saisir l\'identifiant d\'un client:')

    id_input = st.sidebar.selectbox('Selectionner l\'ID du client',tuple(i for i in liste_id))

    if id_input == '': #lorsque rien n'a été saisi
        st.sidebar.text("**Vous devez renseigner l'ID du client**")

    elif (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API

        #Appel de l'API :         
        ID = int(id_input)
        x = df[df['SK_ID_CURR'] == ID]
        
       

        with st.spinner('Chargement du score du client...'):
           pred,proba = api_result(x)

           if pred == 1:
               etat = 'client à risque'
           else:
               etat = 'client peu risqué'
           proba_1 = proba
           #proba_0 = 1 - proba 

           #affichage de la prédiction
           chaine = 'Prédiction : **' + etat +  '** la probabilité que ça soit un client risqué est de **' + str(round(proba_1*100)) + '%**'
        st.subheader("")
        st.markdown(chaine)
        
        st.markdown("# Explication du score de la prédiction #")
        
        #affichage de l'explication du score
        with st.spinner('Chargement des détails de la prédiction...'):
            explanation_pos,explanation_neg = model_local_interpretation(ID, df, model,path_explainer,col_for_explaination)
            explanation_pos = fonction_comparaison(explanation_pos,df,ID,df_train)
            explanation_neg = fonction_comparaison(explanation_neg,df,ID,df_train)
        st.markdown("### Top 6 des Caractéristiques qui contribue à l'accord du prêt ###")
        st.markdown(df_chain_explain(explanation_pos,cat = "positif"),unsafe_allow_html=True)
        #graphe
        #Affichage des graphes    
        graphes_streamlit(explanation_pos, cat = 'positif')

        st.subheader("Définition des groupes")
        st.markdown("\
        \n\
        * Client : la valeur pour le client considéré\n\
        * En Règle : valeur moyenne pour l'ensemble des clients en règle\n\
        * En Défaut : valeur moyenne pour l'ensemble des clients en défaut\n\
        ")

        st.markdown("### Top 6 des Caractéristiques qui s'oppose à l'accord du prêt ###")
        st.markdown(df_chain_explain(explanation_neg , cat = "negatif"), unsafe_allow_html=True)
        #graphe
        #Affichage des graphes    
        graphes_streamlit(explanation_neg,cat = 'negatif')

        st.subheader("Définition des groupes")
        st.markdown("\
        \n\
        * Client : la valeur pour le client considéré\n\
        * En Règle : valeur moyenne pour l'ensemble des clients en règle\n\
        * En Défaut : valeur moyenne pour l'ensemble des clients en défaut\n\
        ")
        
        #Modifier le profil client en modifiant une valeur
        #st.subheader('Modifier le profil client')
        st.sidebar.header("Modifier le profil client")
        st.sidebar.markdown('Cette section permet de modifier une des valeurs les plus caractéristiques du client et de recalculer son score')
        features = explanation_neg['feature'].values.tolist()
        liste_features = tuple([''] + features)
        feature_to_update = ''
        feature_to_update = st.sidebar.selectbox('Quelle caractéristique souhaitez vous modifier', liste_features)

        x_update = df[df['SK_ID_CURR'] == ID]
        #st.write(df.head())

        if feature_to_update != '':
            default_value = explanation_neg[explanation_neg['feature'] == feature_to_update]['customer_values'].values[0]

            min_value = int(df[feature_to_update].values.min())
            max_value = int(df[feature_to_update].values.max())
            print("default_value",default_value)
            print("min_value",min_value)
            print("max_value",max_value)

            if (min_value, max_value) == (0,1): 
                step = float(1)
            else :
                step = int((max_value - min_value) / 20)
            print("step",step)
            update_val = st.sidebar.slider(label = 'Nouvelle valeur',
                min_value = min_value,
                max_value = max_value)

            if update_val != default_value:
                time.sleep(0.5)
                x_update[feature_to_update] = update_val
                update_pred, update_proba = api_result(x_update)
                if update_pred == 1:
                    etat_update = 'client à risque'
                else:
                    etat_update = 'client peu risqué'
                update_proba_1 = update_proba
                #proba_0 = 1 - update_proba 
                chaine = 'Nouvelle prédiction : **' + etat_update +  '** avec **' + str(round(update_proba_1*100)) + '%** de risque de défaut (classe réelle : '+str(pred) + ')'
                st.sidebar.markdown(chaine)
           #faire un dictionnaire de translation des features

    else: 
        st.write('Identifiant non reconnu')
        st.write("exemple d'id client :\n100001\n")

if __name__=='__main__':
    main(path_df, path_train)