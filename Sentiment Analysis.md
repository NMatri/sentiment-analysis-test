## Exercice de Sentiment Analysis
 Le fichier main_script.py contient ma réponse pour l’exercice d'analyse de sentiment. Il s'agit d'un problème de classification binaire balancée.

Le script permet d’entraîner et tester deux modèles de classification. Un modèle qui utilise l'algorithme SVM et un deuxième qui utilise un algorithme de deep learning avec une couche CNN.

### Exécution du script
Le fichier requirements.txt contient la liste de package python nécessaire pour l'exécution du script.

Pour l'exécution du script, il aussi nécessaire de téléchargé le stopwords corpora de nltk en utilisant ces lignes de code:
 ````
 import  nltk
 nltk.download()
````
Il ne reste ensuite que de choisir l'onglet corpora sélectionner stopwords de la liste des différents copora.

Pour afficher le résultat de l'un des modèles, il suffit d'utiliser l'une des commandes suivantes:
````
python main_script.py svm
python main_script.py cnn
````

### Les modèles et les traitements
La création des deux modèles est composé principalement de 6 étapes : 

 1. Chargement des données
 2. Nettoyage des données
 3. Pré-traitement des données
 4. Traitement des données
 5. Entrainement du modèle et tuning des hyperparamètres
 6. Test du modèle

Les premières étapes sont identiques pour les deux modèles. La différence vient en 4ème étape qui consiste à formater les données avant qu'elles soient ensuite injecter dans les algorithmes.

### Modèle de Deep Learning
Dans un premier temps , j'ai commencé par la création du modèle de deep learning. Le modèle est composé d'une couche CNN, suivie d'une couche de MaxPooling permettant la réduction du dimension de l'output de la couche CNN. Après ces deux couches, il y a deux couches connectées. Tout avant la couche CNN, il y a la couche Embedding, qui comme son nom l'indique, réalise l'embedding des mots du corpus au cours de l'entrainement du modèle.

Ce modèle a une précision globale égale à 0.84.  J'ai fait un tuning de paramètres en augmentant le nombre de filtre de la couche CNN et en ajoutant des neurones aux couches connectées, mais la performance n'a pas pu dépassé le 0.84.

Dans la phase de traitement des documents pour ce modèle, j'ai fait la tokenisation des documents et ensuite ajouter du padding aux documents afin de garantir une même longeur pour tous les documents

### Modèle avec SVM
Pour ce modèle j'ai utilisé l'algorithme SVM suite à l'application du TF-IDF à l'ensemble des documents. Ce modèle a permis d'atteindre une performance supérieure au modèle précédent : une précision globale de 0.88. 
