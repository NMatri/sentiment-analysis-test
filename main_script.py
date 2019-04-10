# Import des librairies

import sys
# Librairies de pré-traitements
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

# Modules de traitements
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Modules de modèles de deep learning
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

from sklearn.svm import SVC

# Modules des métriques d'évaluation
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report



def load_data(file_path):
	df = pd.read_csv(file_path, sep='\n', header=-1, names=['text'])
	return df

def label_data(df):
	df_cp = df.copy()
	df_cp['label'] = ['positive' if i < len(df) // 2 else 'negative' for i in range(len(df))]
	return df_cp

def clean_data(df):
	df_cp = df.copy()
	# Eliminer la balise HTML <br>
	df_cp['text'] = df_cp['text'].transform(lambda doc: doc.replace('<br /><br />', ' '))
	# Eliminer les caractères hexadecimaux
	df_cp['text'] = df_cp['text'].transform(lambda doc: re.sub(r'[^\x00-\x7f]',r'', doc))
	# Eliminer les urls
	df_cp["text"] = df_cp["text"].transform(lambda doc: re.sub(r'(https://[^\s]+)', ' ', doc))
	# Garder uniquement les mots
	df_cp["text"] = df_cp["text"].transform(lambda doc: re.sub(r'[^\w\s]', ' ', doc))
	# Transformer en minuscule
	df_cp["text"] = df_cp["text"].transform(lambda doc: doc.lower())
	# Eliminer les caratères numériques
	df_cp["text"] = df_cp["text"].transform(lambda doc: re.sub(r'[0-9]+[a-z]*', ' ', doc))
	# Eliminer les caractères dupliqués
	df_cp["text"] = df_cp["text"].transform(lambda doc: re.sub(r'([a-z])\1+', r'\1', doc))
	return df_cp


def remove_stopwords(doc, stopwords):
	doc_words = doc.split(" ")
	doc_words_new = doc_words
	for word in stopwords:
		doc_words_new = list(filter(lambda doc_word: doc_word != word, doc_words_new))
	return ' '.join(doc_words_new)

def preprocessing(df):
	df_cp = df.copy()
	stopWords = stopwords.words('english')
	df_cp["text"] = df_cp["text"].transform(lambda doc: remove_stopwords(doc, stopWords))
	return df_cp

def max_length(corpus_seq):
	max_len = 0
	for seq in corpus_seq:
		if len(seq) > max_len: max_len = len(seq)  
	return max_len

# Preparer les données pour l'ingection dans le modèle
def data_dl_processing(df_train, df_test):
	# Transformer la colonne text en un vecteur numpy
	train = df_train["text"].values 
	test = df_test["text"].values
	# Création du tokenizer
	tokenizer = Tokenizer(num_words=5000)
	# Entrainement du tokenizer sur le corpus d'entrainement
	tokenizer.fit_on_texts(train)
	# Génération des séquences de mots à partir des documents
	train_encoded_data = tokenizer.texts_to_sequences(train)
	test_encoded_data = tokenizer.texts_to_sequences(test)
	max_len = max_length(train_encoded_data) 	# Détermination de la longeur du plus long document
	# Ajout du padding aux séquences afin de garantir une meme longeur 
	padded_train = pad_sequences(train_encoded_data, maxlen=max_len)
	padded_test = pad_sequences(test_encoded_data, maxlen=max_len)
	return padded_train, padded_test

def train_dl_model(X, y):
	model = Sequential()
	model.add(Embedding(5000, 300, input_length=X.shape[1]))
	model.add(Conv1D(filters=200, kernel_size=2, padding="valid", activation="relu", strides=1))
	model.add(GlobalMaxPooling1D())
	model.add(Dense(128, activation='relu', kernel_initializer="normal"))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu', kernel_initializer="normal"))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid', kernel_initializer="normal"))
	optimizer = Adam(0.001)
	model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
	model.fit(X, y, epochs=5, batch_size=128, verbose=2)
	return model

def data_svm_processing(df_train, df_test):
	train = df_train['text'].values
	test = df_test['text'].values
	vectorizer = TfidfVectorizer(max_features=10000)
	X_train = vectorizer.fit_transform(train)
	X_test = vectorizer.transform(test)
	return X_train, X_test

def train_svm_model(X, y):
	svm_clf = SVC(C=1000, kernel='rbf')
	svm_clf.fit(X, y)
	return svm_clf


if __name__=="__main__":

	model_name = sys.argv[1]
	if model_name not in ['cnn', 'svm']:
		raise ValueError('model name must be either cnn or svm')

	# Charger les données
	print('Chargement des données en cours') 
	df_train = load_data('data/full_train.txt')
	df_test = load_data('data/full_test.txt')


	# Considérer seulement un échantillon
	if model_name == 'cnn':
		df_train_pos = df_train.iloc[0:2500,:]
		df_train_neg = df_train.iloc[len(df_train) // 2 : len(df_train) // 2 + 2500,:]
		df_train = pd.concat([df_train_pos, df_train_neg])

	# Ajouter une colonne contenant le label des documents
	df_train_label = label_data(df_train)
	df_test_label = label_data(df_test)
	
	# Changer l'ordre les lignes de façon aléatoire
	df_train_label = df_train_label.sample(frac=1).reset_index(drop=True)
	df_test_label = df_test_label.sample(frac=1).reset_index(drop=True)

	# Nettoyer les documents
	print('Nettoyage des données en cours')
	df_train_net = clean_data(df_train_label)
	df_test_net = clean_data(df_test_label)

	# Appliquer les pré-traitements
	print('Pré-traitement des données en cours')
	df_train_prepros = preprocessing(df_train_net)
	df_test_prepros = preprocessing(df_test_net)

	# Transformer la colonne label en type category 
	df_train_label['label'] = df_train_label['label'].astype('category')
	df_test_label['label'] = df_test_label['label'].astype('category')
	# Encoder les colonnes label
	label_en = LabelEncoder()
	y_train = label_en.fit_transform(df_train_label['label'])
	y_test = label_en.transform(df_test_label['label'])
	# Renvoie les classes en respectant l'ordre d'encodage
	classes = label_en.classes_

	# Processing des données et entrainement du modèle
	print('Traitement des données et entrainement du modèle en cours')
	if model_name == 'cnn':
		train_pros, test_pros = data_dl_processing(df_train_prepros, df_test_prepros)
		model = train_dl_model(train_pros, y_train)
		y_pred = model.predict_classes(test_pros)
	else:
		train_pros, test_pros = data_svm_processing(df_train_prepros, df_test_prepros)
		model = train_svm_model(train_pros, y_train)
		y_pred = model.predict(test_pros)

	# Afficher le result de test du modèle
	print("classification report: \n", classification_report(y_test, y_pred, target_names=classes))
	acc = accuracy_score(y_test, y_pred)
	print("accuracy score: %.2f" % acc)

