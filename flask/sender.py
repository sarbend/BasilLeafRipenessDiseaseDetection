#sends an image lololol
import requests

files={}
files["bruh"]="brudi"
files["image"]=open("funny.jpg",'rb')
url=input("where u sending? ")
#ez default option shortcut
if url=="a":
    url="127.0.0.1:5000"
#req=requests.post("http://"+url,files=files)
req=requests.Request("POST", "http://"+url, files=files)
reqp=req.prepare()
print(reqp.body)