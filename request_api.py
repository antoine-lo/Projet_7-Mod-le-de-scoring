#Appel de l'API : 
import pandas as pd
from Utils.conf import path_test
import requests
import json
from os import environ, path
from dotenv import load_dotenv


basedir = path.abspath(path.dirname(_file_))
load_dotenv(path.join(basedir, '.env'))

def api_result(x):

  #  url = "http://127.0.0.1:5000/api/"
  url=environ.get("url")


    j_data = x.to_json()
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=j_data, headers=headers)
    print("voici r :",r,"voici r_text :", r.text)

    result = json.loads(r.text)
    print('result prediction :',result["prediction"])

    return result["prediction"],result["probabilite"]
