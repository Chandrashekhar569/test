from flask import Flask,request,rander_templates
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

app = Flask(__init__)

# Route for home page 

@app.route('/')
def home_page():
    return rander_templates(home.html)


@app.route('/predict', methods=['GET','POST'])
def predict_data():
    

if __init__=="__main__":
    aap.run()

