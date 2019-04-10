## Exercice de Sentiment Analysis
 Le fichier main_script.py contient ma réponse pour l’exercice d'analyse de sentiment. Il s'agit d'un problème de classification binaire balancée.

Le script permet d’entraîner et tester deux modèles de classification. Un modèle qui utilise l'algorithme SVM et un deuxième qui utilise un algorithme de deep learning avec une couche CNN.

### Exécution du script
Le fichier requirements.txt contient la liste de packages python nécessaires pour l'exécution du script. Il faut aussi télécharger le stopwords corpora de nltk en utilisant ces lignes de code:
 ````
 import  nltk
 nltk.download()
````
Il ne reste ensuite que de choisir l'onglet corpora et sélectionner stopwords de la liste des différents corpora.

Pour afficher le résultat des modèles, il suffit d'utiliser les commandes suivantes:
````
python main_script.py svm
python main_script.py cnn
````

### Les modèles et les traitements
La création des deux modèles est composée principalement de 6 étapes : 

 1. Chargement des données
 2. Nettoyage des données
 3. Pré-traitement des données
 4. Traitement des données
 5. Entrainement du modèle et tuning des hyperparamètres
 6. Test du modèle

Les premières étapes sont identiques pour les deux modèles. La différence vient en 4ème étape qui consiste à formater les données avant qu'elles soient ensuite injectées dans les algorithmes.

### Modèle de Deep Learning
Dans un premier temps , j'ai commencé par la création du modèle de deep learning. Le modèle est composé d'une couche CNN, suivie d'une couche de MaxPooling permettant la réduction des dimensions de l'output de la couche CNN. Après ces deux couches, il y a deux couches connectées. En tete de l'ensemble de ces couches, il y a la couche Embedding, qui comme son nom l'indique, réalise l'embedding des mots du corpus lors de l'entrainement du modèle.

Ce modèle a une précision globale égale à 0.84.  J'ai fait un tuning de paramètres en augmentant le nombre de filtres de la couche CNN et en ajoutant des neurones aux couches connectées, mais la performance n'a pas pu dépassé 0.84.

### Modèle avec SVM
Pour ce modèle j'ai utilisé l'algorithme SVM suite à l'application du TF-IDF à l'ensemble des documents. Ce modèle a une performance supérieure au modèle précédent : une précision globale de 0.88. 

### Explication des différents choix 

#### Les données

Pour faire entrainer le modèle de DL, j'ai considéré un échantillon des données d'entrainement parce que l'entrainement avec la totalité de la dataset prend beaucoup de temps.

#### Nettoyage des données

Le choix des différents techniques de nettoyage a été fait suite à l'application de la tokenisation aux documents. L'analyse du résultat de la tokenisation sur le text brute m'a permis de voir la présence de balise, des urls etc.

#### Pré-traitements

Le choix de l'elimination des stopwords est basé sur le fait que les documents sont longs et ils contiennent beaucoup de stopwords. De plus, ces mots n'ont pas de valeur sémantique. Donc, ça va permettre de réduire la taille des documents sans perdre de l'information.

#### Les algorithmes

J'ai choisit d'utiliser les algorithmes SVM et deep learning parce qu'ils ont été déja utilisés pour cette problématique et ils permettent d'atteindre de très bonnes performances.

Les hyperparamètres de chacun des algorithmes ont été choisi après tuning et test de différents valeurs pour chaque hyperparamètres.

#### Les traitements

**Deep learning**
Les traitements appliqués aux données sont:

 1. Tokenisation : transformer chaque document en liste de mots en considérant seulement les 5000 mots les plus fréquents. Le choix de 5000 a été fait dans le but de minimiser les dimensions des matrices d'input au modèle tout en gardant une bonne performance de classification. 
 2. Padding : ajouter du padding à chaque de document de sorte que l'emsemble des documents aient la meme longeur.
 3. Embedding : transformer les mots en vecteurs en tenant compte du contexte. Dans notre cas, l'embedding est fait avec la couche du réseaux de neuronnes. Ce choix est due au fait que le résultat de classification a été légèrement meilleur par rapport à l'utilisation du word2vec sur le corpus.

 **SVM**
Le TF-IDF permet de transformer les documents en vecteurs sparse (une matrice dans le cas du DL) contenant des coefficents indiquant l'importance de chacun de leurs mots. Ce qui m'a permis ensuite de considérer l'ensemble du corpus pour l'entrainement et avoir un modèle avec une performance meilleur que celle du premier.