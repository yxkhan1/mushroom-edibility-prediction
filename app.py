import numpy as np
import pandas as pd
import sklearn
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app= Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

df=pd.read_csv("mushrooms.csv")

alpha={'?':0,'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,
       'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z':26}

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["GET","POST"])
def predict():
    features = [x for x in request.form.values()]
    features=pd.DataFrame(features)
    # print(features)
    for i in features:
        features[i]=features[i].map(alpha)
    features=np.asarray([features[0]])
    prediction = model.predict(features)
    if prediction==1:
        pred='Edible'
    else:
        pred='Poisonous'
    return render_template("index.html", prediction_text = "The mushroom is {}".format(pred))

if __name__ == "__main__":
    flask_app.run(debug=True)