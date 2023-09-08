from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# load the model from disk
filename = 'naive_bayes_model.pkl'
naive_bayes_model = pickle.load(open(filename, 'rb'))
logistic_regression_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
CV_naive=pickle.load(open('transform.pkl', 'rb'))
CV_logistic=pickle.load(open('transform_logistic.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        Message = request.form['message'].lower()
        data = [message]
        vect_naive = CV_naive.transform(data).toarray()
        vect_logistic = CV_logistic.transform(data).toarray
        # for naive bayes classifier
        my_nb_prediction = naive_bayes_model.predict(vect_naive)
        nb_percentage = naive_bayes_model.predict_proba(vect_naive)
        if my_nb_prediction==1:
            nb_percentage = nb_percentage[0][1]
        else:
            nb_percentage = nb_percentage[0][0]
        #for logistic regression classifier
        my_lg_prediction = logistic_regression_model.predict(vect_logistic)
        lg_percentage = logistic_regression_model.predict_proba(vect_logistic)
        if my_lg_prediction==1:
            lg_percentage = lg_percentage[0][1]
        else:
            lg_percentage = lg_percentage[0][0]
        return render_template('result.html',
                     message=message,
                     my_nb_prediction = my_nb_prediction,
                     nb_percentage=nb_percentage,
                     my_lg_prediction = my_lg_prediction,
                     lg_percentage=lg_percentage
                     )
if __name__=='__main__':
    app.run(debug=False)
