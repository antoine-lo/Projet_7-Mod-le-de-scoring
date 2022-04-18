#Appel de l'API : 
import requests
import json
import os
from dotenv import load_dotenv


#basedir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
#load_dotenv(os.path.join(basedir, '.env'))
load_dotenv()

def api_result(x):

    #  url = "http://127.0.0.1:5000/api/"
    url=os.environ.get("url")

    print(url)

    j_data = x.to_json()
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=j_data, headers=headers)
    print("voici r :",r,"voici r_text :", r.text)

    result = json.loads(r.text)
    print('result prediction :',result["prediction"])

    return result["prediction"],result["probabilite"]
