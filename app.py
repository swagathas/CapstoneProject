from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if (request.method == 'POST'):
        
        user_input=[str(x) for x in request.form.values()]
        user_input=user_input[0]
        #print(user_input)
        pickled_tfidf_vectorizer = pickle.load(open('tfidf_model.pkl','rb'))
        pickled_model = pickle.load(open('Logistic_regression_Model.pkl','rb'))
        pickled_user_recommendation_model = pickle.load(open('User_based_recommendation_model.pkl','rb')) 
        pickled_cleaned_data = pickle.load(open('cleaned_df.pkl','rb'))

        user_top_20 = User_based_recommendation_model.loc[input].sort_values(ascending = False)[:20]
        user_top_20 = pd.DataFrame(user_top_20)
        user_top_20.reset_index(inplace = True)

        top_20_sentiment = pd.merge(user_top_20, pickled_cleaned_data, on = ['name'])
        top20_tfidf = tf_idf_model.transform(top_20_sentiment["user_review"])
        top20_pred = LR_sentiment_model.predict(top20_tfidf)

        top_20_sentiment['pred_sentiment_score'] = top20_pred

        sentiment_score = top_20_sentiment.groupby(['name'])['pred_sentiment_score'].agg(['sum', 'count']).reset_index()
        sentiment_score['percentage'] = round((100*sentiment_score['sum'] / sentiment_score['count']),2)
        sentiment_score = sentiment_score.sort_values(by = 'percentage',ascending = False)

        output_display=sentiment_score['name'].head(5)

        output = output_display.to_list()
        output.insert(0,"***")
        output="***\t \t***".join(output)
        #print(output)
        return render_template('index.html', prediction_text='Top 5 recommendations are- {}'.format(output))
    else :
        return render_template('index.html')


if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)

