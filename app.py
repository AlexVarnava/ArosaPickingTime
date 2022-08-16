#!/usr/bin/env python
# coding: utf-8

# In[6]:


from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as dp
import pickle
import numpy as np

app = Flask(__name__)


# In[7]:


model = load_model('Arosa_pick_time_lightgbm_tune40')
cols = ['Номенклатура', 
        'ВидНоменклатуры',
        'ЯчейкаУчастокСклада',
        'Ячейка',
        'Ряд',
        'Ярус',
        'Место',
        'Исполнитель',
        'ЕдиницаИзмерения',
        'НоменклатураТипНоменклатуры',
        'НоменклатураВес',
        'КоличествоФакт',
        'ВесСтроки']


# In[8]:


@app.route("/")
def home():
    #return "<h1>Здесь будет модель предсказаний</h1>"
    return render_template('home.html')


# In[9]:


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data = data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    print(prediction)
    return render_template('home.html', pred='Магическое предсказание: {}'.format(prediction))
    #return prediction


# In[11]:


#app.run(host='0.0.0.0', port=50555, debug=True)
app.run(debug=True)


# In[ ]:




