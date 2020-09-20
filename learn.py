import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier




if __name__ == "__main__":
	np.random.seed(555)
	
	#Get DATA
	data_folder = "dataset/splited_data"
	dataset = load_files(data_folder, shuffle=True)
	print("n_samples: %d" % len(dataset.target))
	
	#BEGIN DATA PRE-PROCESSING
	 
	#data to lowercase
	dataset.data = [entry.lower() for entry in dataset.data]

	#TODO :Tokenization
	#TODO :Remove stop words
	#TODO :Stemming/Lemmenting
	
	#END DATA PRE-PROCESSING
	
	# split the dataset in training and test set:
	docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.10, random_state=None)
	
	#Vectorizer
	count_vect = CountVectorizer()
	docs_train_counts = count_vect.fit_transform(docs_train)
	docs_test_counts = count_vect.transform(docs_test)
	
	#TF IDF
	tfidf_transformer = TfidfTransformer()
	docs_train_tfidf = tfidf_transformer.fit_transform(docs_train_counts)
	docs_test_tfidf = tfidf_transformer.transform(docs_test_counts)
	
	# fit the training dataset on the NB classifier
	Naive = naive_bayes.MultinomialNB(alpha=0.26)
	Naive.fit(docs_train_tfidf,y_train)
	# predict the labels on validation dataset
	predictions_NB = Naive.predict(docs_test_tfidf)
	print("Naive Bayes Accuracy Score -> ",np.mean(predictions_NB == y_test))
	
	#Classifier SVM
	SVM = svm.SVC(C=5, kernel='linear')
	SVM.fit(docs_train_tfidf,y_train)
	#predict
	predictions_SVM = SVM.predict(docs_test_tfidf)
	print("SVM Accuracy Score -> ",np.mean(predictions_SVM == y_test))
	
	#Classifier SGD
	SGD = SGDClassifier(loss="modified_huber", penalty="l2", max_iter=300)
	SGD.fit(docs_train_tfidf,y_train)
	#predict
	predictions_SGD = SGD.predict(docs_test_tfidf)
	print("SGD Accuracy Score -> ",np.mean(predictions_SGD == y_test))
	
	#predict new doc with SGD
	docs_new = ['oh seriously, you are a true dick ! fuck you', "i'm so proud of him ! good boy ","i really hope we can save all of us, but i can't do it alone","joy and hapiness is it the same thing ?"]
	X_new_counts = count_vect.transform(docs_new)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	predicted = SGD.predict(X_new_tfidf)
	for doc, category in zip(docs_new, predicted):
		print('%r => %s' % (doc, dataset.target_names[category]))
	
	
	
