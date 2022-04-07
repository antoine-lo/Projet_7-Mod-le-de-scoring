import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
#import lime
import time
#from lime import lime_text
#import lime.lime_tabular
import model
from conf import path_test,path_pipeline_obj,threshold


### import path data model
path_df = path_test
path_model = path_pipeline_obj

#Load Dataframe
df = pd.read_csv(path_test)

id = "100001"

#Load model
model = PipelinePredictor(path_pipeline_obj)

### fonction
def load_df(path):
    df = pd.read_csv(path)
    df = df.loc[:,~df.columns.str.startswith('Unnamed')]
    return df

def pret_prediction(ID, df, threshold, model ):
    '''Renvoie la prediction a partir du modele ainsi que les probabilites d\'appartenance à chaque classe'''

    ID = int(ID)
    x = df[df['SK_ID_CURR'] == ID]

    prediction, proba =  model.predict(x)

    print("prediction :",prediction," proba :",proba)

    if proba[0][1] <= threshold:
        pred = 0
    else :
        pred = 1
    
    return pred, proba[0][1] 

def pret_prediction_flask(ID, df, model,threshold):
    '''Fonction de prédiction utilisée par l\'API flask :
    a partir de l'identifiant et du jeu de données
    renvoie la prédiction à partir du modèle'''

    ID = int(ID)
    x = df[df['SK_ID_CURR'] == ID]

    prediction, proba = model.predict(x)
    
    if proba[0][1] <= threshold:
        pred = 0
    else :
        pred = 1
    
    return pred, proba[0][1] 

def pret_prediction_update_flask(ID, df, feature, value, model,threshold):
    '''Renvoie la prédiction à partir d\'un vecteur x'''
    ID = int(ID)
    x = df[df['SK_ID_CURR'] == ID]

    #update feature
    x.loc[feature] = value
    
    prediction, proba = model.predict(x)

    print(proba)
    
    if proba <= threshold:
        pred = 0
    else :
        pred = 1
    
    return pred, proba

def main(path,id, df, threshold, model):
    load_df(path)
    pred,proba = pret_prediction(id, df, threshold, model )

    return pred,proba


if __name__=="__main__":
    pred,proba = main(path_df,id, df, threshold, model)

    print("la prediction de la classe est : ",pred," la proba associée est :",proba)

"""

def clean_map(string):
    '''nettoyage des caractères de liste en sortie de LIME as_list'''
    signes = ['=>', '<=', '<', '>']
    for signe in signes :
        if signe in string :
            signe_confirme = signe
        string = string.replace(signe, '____')
    string = string.split('____')
    if string[0][-1] == ' ':
        string[0] = string[0][:-1]

    return (string, signe_confirme)

def interpretation(ID, df, model, sample=False):
    '''Fonction qui fait appel à Lime à partir du modèle de prédiction et du jeu de données'''
    #préparation des données
    print('\n\n\n\n======== Nouvelle Instance d\'explicabilité ========')
    start_time = time.time()
    ID = int(ID)
    class_names = ['OK', 'default']
    df_full = df.copy()
    
    x = df[df['SK_ID_CURR'] == ID]
    
    print('ID client: {}'.format(ID))
    
    print('Temps initialisation : ', time.time() - start_time)
    start_time = time.time()


    #si on souhaite travailler avec un volume réduit de données    
    if sample is True :
        df_reduced = df[df['SK_ID_CURR']==int(ID)]
        df = pd.concat([df_reduced, df.sample(2000, random_state=20)], axis=0)
        del df_reduced

    #fin de préparation des données
    x = x.drop(['SK_ID_CURR', 'LABELS'], axis=1)
    df = df.drop(['SK_ID_CURR', 'LABELS'], axis=1)

    #création de l'objet explainer
    import_explainer = False
    if import_explainer is True:
        print('import explainer true')
        with open(path_explainer, 'rb') as f:
            explainer = dill.load(f)
    else:    
        print('import explainer false')
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data = np.array(df.sample(int(0.1*df.shape[0]), random_state=20)),
            feature_names = df.columns,
            training_labels = df.columns.tolist(),
            verbose=1,
            random_state=20,
            mode='classification')

        #with open(path_explainer, 'wb') as f:
        #    dill.dump(explainer, f)


    print('Temps initialisation explainer : ', time.time() - start_time)
    start_time = time.time()

    #explication du modèle pour l'individu souhaité

    exp = explainer.explain_instance(data_row = x.sort_index(axis=1).iloc[0:1,:].to_numpy().ravel(),
        predict_fn = )

    print('Temps instance explainer : ', time.time() - start_time)
    start_time = time.time()

    #traitement des données et comparaison
    fig = exp.as_pyplot_figure()
    #exp.show_in_notebook(text=False)
    df_map = pd.DataFrame(exp.as_list())
    print(df_map)

    df_map['feature'] = df_map[0].apply(lambda x : clean_map(x)[0][0])
    df_map['signe'] = df_map[0].apply(lambda x : clean_map(x)[1])
    df_map['val_lim'] = df_map[0].apply(lambda x: clean_map(x)[0][-1])
    #df_map['ecart'] = df_map[0].apply(lambda x: clean_map(x)[0][-1])
    df_map['ecart'] = df_map[1]

    df_map = df_map[['feature', 'signe', 'val_lim', 'ecart']]
    #global
    df_map['contribution'] = 'normal'
    df_map.loc[df_map['ecart']>=0, 'contribution'] = 'default'
    
    df_map['customer_values'] = [x[feature].mean() for feature in df_map['feature'].values.tolist()]
    df_map['moy_global'] = [df_complet[feature].mean() for feature in df_map['feature'].values.tolist()]
    #clients en règle
    df_map['moy_en_regle'] = [df_complet[df_complet['LABELS'] == 0][feature].mean() for feature in df_map['feature'].values.tolist()]
    #clients en règle
    df_map['moy_defaut'] = [df_complet[df_complet['LABELS'] == 1][feature].mean() for feature in df_map['feature'].values.tolist()]
    #20 plus proches voisins
    index_plus_proches_voisins = nearest_neighbors(x, df_complet, 20)
    df_map['moy_voisins'] = [df_complet[df_complet['Unnamed: 0'].isin(index_plus_proches_voisins)][feature].mean() for feature in df_map['feature'].values.tolist()]

    print('Temps calcul données comparatives : ', time.time() - start_time)
    start_time = time.time()
    df_map = pd.concat([df_map[df_map['contribution'] == 'default'].head(3),
        df_map[df_map['contribution'] == 'normal'].head(3)], axis=0)

    return df_map.sort_values(by='contribution')


def df_explain(df):
    '''Ecrit une chaine de caractéres permettant d\'expliquer l\'influence des features dans le résultat de l\'algorithme '''

    chaine = '##Principales caractéristiques discriminantes##  \n'
    df_correspondance = pd.DataFrame(columns=['Feature','Nom francais'])
    for feature in df['feature']:

        chaine += '### Caractéristique : '+ str(feature) + '('+ correspondance_feature(feature) +')###  \n'
        chaine += '* **Prospect : **'+ str(df[df['feature']==feature]['customer_values'].values[0])
        chaine_discrim = ' (seuil de pénalisation : ' + str(df[df['feature']==feature]['signe'].values[0])
        chaine_discrim +=  str(df[df['feature']==feature]['val_lim'].values[0])

        if df[df['feature']==feature]['contribution'].values[0] == 'default' :
            chaine += '<span style=\'color:red\'>' + chaine_discrim + '</span>  \n' 
        else : 
            chaine += '<span style=\'color:green\'>' + chaine_discrim + '</span>  \n' 

        #chaine += '* **Clients Comparables:**'+str(df[df['feature']==feature]['moy_voisins'].values[0])+ '  \n'
        #chaine += '* **Moyenne Globale:**'+str(df[df['feature']==feature]['moy_global'].values[0])+ '  \n'
        #chaine += '* **Clients réguliers :** '+str(df[df['feature']==feature]['moy_en_regle'].values[0])+ '  \n'
        #chaine += '* ** Clients avec défaut: **'+str(df[df['feature']==feature]['moy_defaut'].values[0])+ '  \n'
        #chaine += ''
        df_correspondance_line = pd.DataFrame(data = np.array([[feature, correspondance_feature(feature)]]), columns = ['Feature', 'Nom francais'])
        #df_correspondance_line = pd.DataFrame(data = {'Feature' : feature, 'Nom francais' : correspondance_feature(feature)})
        df_correspondance = pd.concat([df_correspondance, df_correspondance_line], ignore_index=True)
    return chaine, df_correspondance


def nearest_neighbors(x, df, n_neighbors):
    '''Determine les plus proches voisins de l\'individu x 
    considere a partir d\'un KDTree sur 5 colonnes représentatives de caractéristiques intelligibles
    Renvoie en sortie les indices des k plus proches voisins'''
    cols = ['DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'CODE_GENDER_F', 'CREDIT_TERM', 'CREDIT_INCOME_PERCENT']
    tree = pickle.load(open(path_kdtree, 'rb'))
    dist, ind = tree.query(np.array(x[cols]).reshape(1,-1), k = n_neighbors)
    return ind[0]

def correspondance_feature(feature_name):
    '''A partir du nom d\'une feature, trouve sa correspondance en français'''
    df_correspondance = pd.read_csv(path_correspondance_features)
    df_correspondance['Nom origine'] = df_correspondance['Nom origine'].str[1:]
    try:
        return df_correspondance[df_correspondance['Nom origine'] == feature_name]['Nom français'].values[0]
    except:
        print('correspondance non trouvée')
        return feature_name

def graphes_streamlit(df):
    '''A partir du df, affichage un subplot de 6 graphes représentatif du client comparé à d'autres clients sur 6 features'''
    f, ax = plt.subplots(2, 3, figsize=(10,10), sharex=False)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    
    i = 0
    j = 0
    liste_cols = ['Client', 'Moyenne', 'En Règle', 'En défaut','Similaires']
    for feature in df['feature']:

        sns.despine(ax=None, left=True, bottom=True, trim=False)
        sns.barplot(y = df[df['feature']==feature][['customer_values', 'moy_global', 'moy_en_regle', 'moy_defaut', 'moy_voisins']].values[0],
                   x = liste_cols,
                   ax = ax[i, j])
        sns.axes_style("white")

        if len(feature) >= 18:
            chaine = feature[:18]+'\n'+feature[18:]
        else : 
            chaine = feature
        if df[df['feature']==feature]['contribution'].values[0] == 'default':
            chaine += '\n(pénalise le score)'
            ax[i,j].set_facecolor('#ffe3e3') #contribue négativement
            ax[i,j].set_title(chaine, color='#990024')
        else:
            chaine += '\n(améliore le score)'
            ax[i,j].set_facecolor('#e3ffec')
            ax[i,j].set_title(chaine, color='#017320')
            
       
        if j == 2:
            i+=1
            j=0
        else:
            j+=1
        if i == 2:
            break;
    for ax in f.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)
    if i!=2: #cas où on a pas assez de features à expliquer (ex : 445260)
        #
        True
    st.pyplot()

    return True

"""


