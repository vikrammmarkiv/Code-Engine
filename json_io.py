#!flask/bin/python
import sys
from flask import Flask, render_template, request, redirect, Response
import random, json
from flask_cors import CORS
import spacy

app = Flask(__name__)

import pandas as pd
import socket
import os
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#creating tokens
data = open("data.txt","r")
wdata = open("wdata.txt","w")
print("1")
for d in data:
    tokenized_sent = word_tokenize(d)
    for t in tokenized_sent:
        wdata.write(t+"\n")
wdata.close()		
#reading dataset
csvfile = pd.read_csv("data2.csv")
print("2")

#processsing the data for ml
y = csvfile.label
X_train, X_test, y_train, y_test = train_test_split(csvfile["text"],y,test_size=0.33,random_state=33)
count_vectorizer = CountVectorizer()
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)
X_train=X_train.fillna("")
X_test=X_test.fillna("")
print("3")
#creating ml model for classification
nb_classifier = MultinomialNB(alpha=0.1)
nb_classifier.fit(count_train,y_train)
print("4")
en_nlp = spacy.load('en')

@app.route('/receiver', methods = ['POST'])
def worker():
	print("1")
	# read json + reply
	print("5")
	query0 = request.form.get("que")
	doc = en_nlp(unicode(query0))
	print(query0)
	query=word_tokenize(query0)
	count_user = count_vectorizer.transform(query)
	result = nb_classifier.predict(count_user)
	print (result)
	i=0
	j=0
	dic={}
	dicd={}
	liso=[]
	dic2=dict((y,x) for (x,y) in zip(result,query))
	for x,y in zip(result,query):
		if(x=='data'):
			i=i+1
			dicd["d"+str(i)]=y
			dicd["op"+str(i)]=getop(y,doc,dic2)
		elif(x=='op'):
			liso.append(y)
			j+=1;
		elif(x=='o'):
			continue
	print(dicd)
	dic['operation']=liso
	dic['data']=dicd
	dic['data_count']=i
	print (dic)
	resul = json.dumps(dic)
	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client_socket.connect(("localhost", 5050))
	client_socket.send(resul+"\n")
	client_socket.close()
	return resul

@app.after_request
def after_request(response):
	response.headers.add('Access-Control-Allow-Origin', '*')
	response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
	response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
	return response
	
def getop(data,doc,dic2):
	for word in doc:
		if (word.text==data):
			for parent in word.ancestors:
				#print("ances "+word.text+" - "+parent.text)
				# Check for an "item" entity
				if  dic2[parent.text] == "op":
					return parent.text
			return None
	
	

	
if __name__ == '__main__':
	# run!
	print("6")
	app.run()
