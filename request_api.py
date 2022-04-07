#Appel de l'API : 
import pandas as pd
from Utils.conf import path_test
import requests
import json


"""df = pd.read_csv(path_test)
url = 'http://127.0.0.1:5000/api/'
id_input = "100001
ID = int(id_input)
x = df[df['SK_ID_CURR'] == ID]
"""
def api_result(x):

    url = "http://127.0.0.1:5000/api/"

    j_data = x.to_json()
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=j_data, headers=headers)
    print("voici r :",r,"voici r_text :", r.text)

    result = json.loads(r.text)
    print('result prediction :',result["prediction"])

    return result["prediction"],result["probabilite"]
