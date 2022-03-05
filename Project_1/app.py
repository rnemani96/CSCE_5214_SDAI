from flask import Flask,render_template,url_for,request,current_app,send_from_directory
import pandas as pd 
import pickle
import re
import nltk 
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
from nltk.stem import WordNetLemmatizer


# app=Flask(__name__,template_folder='/Template')
app = Flask(__name__)
app.secret_key = "s3cr3t"
app.debug = False
app._static_folder = os.path.abspath("templates/static/")
# @app.route('/')
# def home():
# 	# return current_app.send_static_file('index.html')
# 	return send_from_directory('Template', 'index.html')

@app.route("/", methods=["GET"])
def index():
    title = "ML Model"
    return render_template("layouts/index.html", title=title)	

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("IMDB_Dataset.csv")
	
	df['R_1'] = (((((df['review'].replace("\r", " ", regex=True)).replace("\n", " ", regex=True)).replace("    ", " ")).replace('"', '',  regex=True)).replace("\t", " ", regex=True))
	df['R_2'] =df['R_1'].str.replace('(<br />|\d+\.)','').str.split().agg(" ".join)
	df['R_3'] = df['R_2'].str.lower()
	df['R_3'] = df['R_3'].replace("'s", "", regex=True)
	punctuation_signs = list("?:!.,;'")
	df['R_4'] = df['R_3']
	for punct_sign in punctuation_signs:
  		df['R_4'] = df['R_4'].str.replace(punct_sign, '',regex=True)
	
	wordnet_lemmatizer = WordNetLemmatizer()
	
	nrows = len(df)
	lemmatized_text_list = []

	for row in range(0, nrows):
		lemmatized_list = []
		text = df.loc[row]['R_4']
		text_words = text.split(" ")
		for word in text_words:
			lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
		lemmatized_text = " ".join(lemmatized_list)
		lemmatized_text_list.append(lemmatized_text)
	df['R_5'] = lemmatized_text_list
	df['R_6'] = df['R_5']
	stop_words = set(stopwords.words("english"))
	for stop_word in stop_words:

		regex_stopword = r"\b" + stop_word + r"\b"
		df['R_6'] = df['R_6'].str.replace(regex_stopword, '')
	sentences=df['R_6']
	le=LabelEncoder()
	df['sentiment']= le.fit_transform(df['sentiment'])
	
	tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
	matrix_count = tfidf.fit_transform(df.R_6)
	X_train, X_test, Y_train, Y_test = train_test_split(matrix_count, df.sentiment, test_size=0.20, random_state=2)
	L_SVC = LinearSVC()
	L_SVC.fit(X_train, Y_train)
	L_SVC.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# joblib.dump(L_SVC, 'Linear_SVC_Model.pkl')
	# Linear_SVC_Model = open('Linear_SVC_Model.pkl','rb')
	# L_SVC = joblib.load(Linear_SVC_Model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = tfidf.fit_transform(data).toarray()
		my_prediction = L_SVC.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(host= '0.0.0.0', port=8080)