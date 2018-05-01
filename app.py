#!flask/bin/python
import sys
from flask import Flask, render_template, request, redirect, Response
import random, json
from flask_cors import CORS
import spacy
from collections import defaultdict
import subprocess	
import re

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
y = csvfile.label1
y2 = csvfile.label2
X_train, X_test, y_train, y_test = train_test_split(csvfile["text"],y,test_size=0.33,random_state=33)
X_train2, X_test2, y_train2, y_test2 = train_test_split(csvfile["text"],y2,test_size=0.33,random_state=33)
count_vectorizer = CountVectorizer()
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)
X_train=X_train.fillna("")
X_test=X_test.fillna("")
print("3")
#creating ml model for classification
nb_classifier = MultinomialNB(alpha=0.1)
nb_classifier.fit(count_train,y_train)
nb_classifier2 = MultinomialNB(alpha=0.1)
nb_classifier2.fit(count_train,y_train2)
flag = 0
order = "default"
print("4")
en_nlp = spacy.load('en')


@app.route('/receiver', methods = ['POST'])
def worker():
	print("1")
	
	# read json + reply
	print("5")
	lineinit = 9
	linefun = 18
	linedata = 10
	lineop=11
	with open('generator/JClass.java', 'r') as file:
		data2 = file.readlines()	
	query0 = request.form.get("que")
	query123 = query0.splitlines()
	for query0 in reversed(query123):
		query0 = query0.replace(',',' and ')
		if '\"' in query0:
			datatype = "string"
		else:
			datatype = "integer"
		query0 = query0.replace('\"','')	
		print(query0)
		query=word_tokenize(query0)
		count_user = count_vectorizer.transform(query)
		result = nb_classifier.predict(count_user)
		print (result)
		result2 = nb_classifier2.predict(count_user)
		print (result2)
		i=0
		j=0
		dic={}
		dicd= defaultdict(list)
		liso=[]
		list1 = query0.split()
		print(list1)
		global order
		order='default'
		neg = "none"
		cmp = "none"
		for i,(x,y) in enumerate(zip(result,query)):
			if(x=='op'):
				query[i]=result2[i]
				list1[i] = result2[i]
				query0 = ' '.join(list1)
			if(x=='order'):
				order=result2[i]
		query2=word_tokenize(query0)
		query=word_tokenize(query0)
		print("***************************")
		result=result.tolist()
		result2=result2.tolist()
		print(result)
		if "result" in query0:
			resindex = query.index("result")
			resop = ""
			for i in range(resindex,len(result)):
				if result[i]=='op':
					resop=query2[i]
					break
			query[resindex]="result_"+resop
			query.pop(i)
			result.pop(i)
			result2.pop(i)
		flag0 = 0
		if "is greater than" in query0:
			resindex = query.index("greater")
			query[resindex]='>'
			flag0=1
		elif "is less than" in query0:
			resindex = query.index("less")
			query[resindex]='<'
			flag0=1
		if "is equal to" in query0:
			resindex = query.index("equal")
			query[resindex]='=='
			flag0=1
		if flag0 == 1:	
			query.pop(resindex-1)
			result.pop(resindex-1)
			result2.pop(resindex-1)
			query.pop(resindex)
			result.pop(resindex)
			result2.pop(resindex)
		
		query0 = ' '.join(query)
		print(query0,query,result)	
		i=0
		doc = en_nlp(unicode(query0))
		dic2=dict((y,x) for (x,y) in zip(result,query))
		dic['condition']="false"
		for x,y in zip(result,query):
			if(x=='data'):
				i=i+1
				dicd[getop(y,doc,dic2)].append(y)
			elif(x=='op'):
				liso.append(y)
				j+=1;
			elif(x=='condition'):
				dic['condition']="true"
			elif(x=='cmp'):
				cmp = y
			elif(x=='negation'):
				neg = "true"
			elif(x=='o'):
				continue	
		if "result of" in query0:
			dicd[getop("result",doc,dic2)].append("result_"+resop)
		print(dicd)
		dic['operations']=liso
		if(len(liso)>1):
			return json.dumps({'output':"Please enter multiple operations in different lines",'code':""})
		dic['data']=dicd
		dic['data_count']=i
		dic['order']=order
		dic['if']=[neg,cmp]
		dic['query']=[query,result]
		print (dic)
		resul = json.dumps(dic)
		lineinit,linefun,linedata,lineop,data2 = codegen(dic,datatype,lineinit,linefun,linedata,lineop,data2)
	output = compile_java()
	print(output)	
	with open('out.java', 'r') as file:
		codefile = file.read()		
	print(codefile)
	resul = json.dumps({'output':output,'code':codefile})
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
			for parent in doc:
				#print("ances "+word.text+" - "+parent.text)
				# Check for an "item" entity
				if  dic2[parent.text] == "op":
					return parent.text
			return None
	
def codegen(nlpdata,datatype,lineinit,linefun,linedata,lineop,data2):		
	dval = nlpdata['data']
	data_count = nlpdata['data_count']
	op = nlpdata['operations']
	order = nlpdata['order']
	condition = "false"
	condition = nlpdata['condition']
	dicif = nlpdata['if']
	dicquery=nlpdata['query']
	datadis = []
	
	
	# and write everything back
	for i in op:	
		if (i=="display" and len(op)==1) or condition == "true" :
			continue
		else:
			data2.insert(lineinit,"\t\tArrayList<Double> "+i+"_data = new ArrayList<Double>();  //List to store data of "+i+" operation\n")
			linefun+=1
			lineop+=1
			linedata+=1	
	r = re.compile("(?!result).")
	r2 = re.compile("(?=result).")
	
	for val in dval:
		if val!=None:			
			m = filter(r.match, dval[val])
			for d in reversed(m):
				if (val == "display" and len(op)==1) or condition == "true" :
					datadis.append(d)
				else:
					data2.insert(linedata,"\t\t"+val+"_data.add((double)"+str(d)+");\n")
					linefun+=1
					lineop+=1
	
	opdic={"addition":general,"max":general,"displaystr":displaystr,"cond":cond,"divide":general,"random":general,"factorial":general,"fibonacci":general,"mod":general,"multiply":general,"palindrome":general,"power":general,"sort":sortf,"subtraction":general,"factor":general,"min":general,"prime":general,"multiple":general,"percentage":general,"vowels":general}
	lined=lineop
	if len(op)==0:	
		if condition == "true":
				linefun,lined = opdic["cond"]([],data2,lineop,linefun,lined,[],dicif,datatype,dicquery)
			
	for o in reversed(op):
		m = filter(r2.match, dval[o])
		if condition == "true":
			linefun,lined = opdic["cond"](o,data2,lineop,linefun,lined,dval[o],dicif,datatype,dicquery)
		else:
			if o!="display" and "display" in op:
				data2.insert(lineop,"\t\t"+"result.add(new String[]{String.valueOf(result_"+o+"),\""+o+"\"});\n\n")
				linefun+=1
				lined+=1
			if len(m)>0 :
				for x in m:
					if o == "display" and len(op)==1:
						datadis.append(x)
			if o == "display" and len(op)==1:
				linefun,lined = opdic[o+"str"](o,data2,lineop,linefun,lined,datadis,datatype)	
			else:
				linefun,lined = opdic[o](o,data2,lineop,linefun,lined)	
				data2.insert(lineinit,"\t\tdouble result_"+o+"=-1;\n")
				linefun+=1
				lineop+=1
				linedata+=1
	
			if len(m)>0 :
				for x in m:
					if o!="display":
						global flag
						flag=1
						data2.insert(lineop,"\t\t"+o+"_data.add("+str(x)+");\n")	
						linefun+=1
						lined+=1
		
		
		
	
	input = []	
	
	for o in reversed(op):
		if o=="display":
			continue
		else:
			with open("generator/operations/"+o, 'r') as file:
				# read a list of lines into data
				data = file.readlines()
			data.reverse()
			input.append(data);
	
	
	for fun in input:
		for lv in fun:
			data2.insert(linefun,lv)
		
	
	with open('out.java', 'w') as file:
		file.writelines(data2)
	return lineinit,linefun,linedata,lineop,data2
	
def compile_java():
	proc = subprocess.check_call(['javac','out.java'])
	proc2 = subprocess.Popen('java out', shell=True, stdout=subprocess.PIPE)
	out, err = proc2.communicate()
	return out
	
def general(op,data2,lineop,linefun,lined):	
	data2.insert(lineop,"\t\t"+"result_"+op+'='+"obj."+op+"("+op+"_data);  //Calling "+op+"() function\n")
	linefun+=1
	lined+=1
	return linefun,lined

def displaystr(op,data2,lineop,linefun,lined,datadis,datatype):
	global flag
	flag=0
	r2 = re.compile("(?=result).")	
	m = filter(r2.match, datadis)
	if len(m)>0 :
				for x in m:
					if x=="result_sort":
						data2.insert(lineop,"\t\tSystem.out.println(sort_data);\n")
					else:
						data2.insert(lineop,"\t\tSystem.out.println("+x+");\n")
					linefun+=1
					lined+=1
					datadis.remove(x)
	if(len(datadis)>0):
		data2.insert(lineop,"\t\tSystem.out.println(\""+' '.join(reversed(datadis))+"\");\n")			
		linefun+=1
		lined+=1
	return linefun,lined

def cond(op,data2,lineop,linefun,lined,d,dicif,datatype,result):
	if len(op)>0:
		if dicif[0] != "true":
			if datatype == "string":
				data2.insert(lineop,"\t\tif(obj."+op+"(\""+d[0]+"\")){\n")	
			else:
				data2.insert(lineop,"\t\tif(obj."+op+"("+d[0]+")){\n")	
		else:
			if datatype == "string":
				data2.insert(lineop,"\t\tif(!obj."+op+"(\""+d[0]+"\")){\n")	
			else:
				data2.insert(lineop,"\t\tif(!obj."+op+"("+d[0]+")){\n")		
	else :
		da=[]
		for i,x in enumerate(result[1]):
			if x=='data':
				da.append(result[0][i])
		if dicif[0] != "true":
			data2.insert(lineop,"\t\tif("+da[0]+dicif[1]+da[1]+"){\n")		
		else:
			data2.insert(lineop,"\t\tif(!"+da[0]+dicif[1]+da[1]+"){\n")	
	global flag
	if flag == 1:
		data2.insert(lineop+3,"\t\t}\n")
	else:
		data2.insert(lineop+2,"\t\t}\n")
		
	linefun+=2
	lined+=2
	return linefun,lined
	
def sortf(op,data2,lineop,linefun,lined):
	global order
	data2.insert(lineop,"\t\t"+"result_"+op+'='+"obj."+op+"("+op+"_data,\""+order+"\");  //Calling "+op+"() function\n")
	linefun+=1
	lined+=1
	return linefun,lined
	

if __name__ == '__main__':
	# run!
	print("6")
	app.run(host='10.0.0.5',port=8080)
