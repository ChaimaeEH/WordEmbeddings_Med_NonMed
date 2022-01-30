from gensim.models import Word2Vec
from gensim.models import FastText
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import fasttext
import pandas as pd
import pickle
import pdb


################################### Fonction communes à toutes les parties #####################################@

# Récuperer uniquement la liste des 10 mots les plus proches (sans le score de similarité)
# fonction qui sera utilisé après
def proches_mots(sims) :
    prochesmots = []
    for i in range (len(sims)):
        prochesmots.append(sims[i][0])
    return prochesmots

# fonction pour chercher les 10 mots les plus similaires de 4 mots donnés ici patient, traitement, maladie et jaune
# puis ensuite ne recuperer que la liste des mots sans leur score de dimilarité
def dix_mots_sans_score(model):
    A = model.wv.most_similar('patient', topn=10)
    B = model.wv.most_similar('traitement', topn=10)
    C = model.wv.most_similar('maladie', topn=10)
    D = model.wv.most_similar('jaune', topn=10)
    A = proches_mots(A)
    B = proches_mots(B)
    C = proches_mots(C)
    D = proches_mots(D)
    return A,B,C,D

# recuperer les coordonnes en 2 dimensions du modele afin de pouvoir le ploter
# on utilise un modèle TSNE pour passer d'une dimension 100 à une dimension 2 pour avoir un graphique 
def coord_2D(model, data): 
    X = model.wv[data]
    tsne = TSNE(n_components=2,verbose = 1) # appeler le modèle avec que 2 composants
    X_tsne = tsne.fit_transform(X) # passer de 100 à 2 
    df = pd.DataFrame(X_tsne, index=data, columns=['x', 'y'])
    return df

'_____________________________________________ Press Corpus ___________________________________________________'

##################################################################################################################
#                                           Word2Vec : CBOW 
##################################################################################################################


# ouvrir le modèle et chercher les mots les plus proches : 
model_CBOW_press = Word2Vec.load("word2vec_CBOW_press.model")


# chercher les 10 plus proches mots sans les scores de similarité de 4 mots 
prochesmots_patients_list,prochesmots_traitement_list,prochesmots_maladie_list,prochesmots_jaune_list = dix_mots_sans_score(model_CBOW_press)

# les mots dont on cherche les mots les plus proches et qu'on va ploter aussi. 
mots_cibles = ['patient','maladie','traitement','jaune']

data_press_liste = model_CBOW_press.wv.key_to_index.keys() # liste de tous les mots du corpus press 
#df = coord_2D(model_CBOW_press, data_press_liste) # coordonnees en 2D de tous les mots du corpus press 

# Pour enregistrer le file la première fois histoire car il prend du temps, puis on aura juste besoin de le charger.
#with open('results_TSNE_press.pkl', 'wb') as file:    
    # A new file will be created
#    pickle.dump(df, file)

### Pour ouvrir le file une fois enregistré  ###
with open('results_TSNE_press.pkl', 'rb') as file:     
    # Call load method to deserialze
    df = pickle.load(file)

# Rechercher dans la variable df les coordonnes x,y des mots cibles et des mots les plus proches 
prochesmots_mots_cibles = df.loc[mots_cibles]
prochesmots_patients = df.loc[prochesmots_patients_list] #patient 
prochesmots_traitement = df.loc[prochesmots_traitement_list] #traitement
prochesmots_maladie = df.loc[prochesmots_maladie_list] #maladie
prochesmots_jaune = df.loc[prochesmots_jaune_list] #jaune

# tracer le graphique des mots proches et des mots cibles 
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(prochesmots_patients['x'], prochesmots_patients['y'], color= 'red', label = 'PM patient')
ax.scatter(prochesmots_traitement['x'], prochesmots_traitement['y'], color= 'green',label = 'PM traitement')
ax.scatter(prochesmots_maladie['x'], prochesmots_maladie['y'], color= 'purple', label = 'PM maladie')
ax.scatter(prochesmots_jaune['x'], prochesmots_jaune['y'], color= 'pink', label = 'PM jaune')
ax.scatter(prochesmots_mots_cibles['x'], prochesmots_mots_cibles['y'], color= 'blue', label = 'mots cibles')


for word, pos in prochesmots_patients.iterrows():
    ax.annotate(word, pos, color = 'red')
for word, pos in prochesmots_traitement.iterrows():
    ax.annotate(word, pos, color = 'green')
for word, pos in prochesmots_maladie.iterrows():
    ax.annotate(word, pos, color = 'purple')
for word, pos in prochesmots_jaune.iterrows():
    ax.annotate(word, pos, color = 'pink')
for word, pos in prochesmots_mots_cibles.iterrows():
    ax.annotate(word, pos, color = 'blue')

plt.title('Modèle Word2Vec : CBOW : CORPUS PRESSE')
plt.grid()
plt.legend()

# enregistrer le graphique
plt.savefig('model_CBOW_press.png')
plt.show()


##################################################################################################################
#                                           Word2Vec : Skip-Gram 
##################################################################################################################

# ouvrir le modèle et chercher les mots les plus proches : 
model_sg_press = Word2Vec.load("word2vec_sg_press.model")


# chercher les 10 plus proches mots sans les scores de similarité de 4 mots 
prochesmots_patients_list,prochesmots_traitement_list,prochesmots_maladie_list,prochesmots_jaune_list = dix_mots_sans_score(model_sg_press)

data_press_liste = model_sg_press.wv.key_to_index.keys() # liste de tous les mots du corpus press 
#df = coord_2D(model_sg_press, data_press_liste) # coordonnees en 2D de tous les mots du corpus press 

# Pour enregistrer le file la première fois histoire car il prend du temps, puis on aura juste besoin de le charger.
#with open('results_TSNE_sg_press.pkl', 'wb') as file:    
    # A new file will be created
#    pickle.dump(df, file)

### Pour ouvrir le file une fois enregistré  ###
with open('results_TSNE_sg_press.pkl', 'rb') as file:     
    # Call load method to deserialze
    df = pickle.load(file)

# coordonnees des proches mots et des mots cibles 
prochesmots_mots_cibles = df.loc[mots_cibles]
prochesmots_patients = df.loc[prochesmots_patients_list] #patient 
prochesmots_traitement = df.loc[prochesmots_traitement_list] #traitement
prochesmots_maladie = df.loc[prochesmots_maladie_list] #maladie
prochesmots_jaune = df.loc[prochesmots_jaune_list] #jaune


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(prochesmots_patients['x'], prochesmots_patients['y'], color= 'red', label = 'PM patient')
ax.scatter(prochesmots_traitement['x'], prochesmots_traitement['y'], color= 'green',label = 'PM traitement')
ax.scatter(prochesmots_maladie['x'], prochesmots_maladie['y'], color= 'purple', label = 'PM maladie')
ax.scatter(prochesmots_jaune['x'], prochesmots_jaune['y'], color= 'pink', label = 'PM jaune')
ax.scatter(prochesmots_mots_cibles['x'], prochesmots_mots_cibles['y'], color= 'blue', label = 'mots cibles')


for word, pos in prochesmots_patients.iterrows():
    ax.annotate(word, pos, color = 'red')
for word, pos in prochesmots_traitement.iterrows():
    ax.annotate(word, pos, color = 'green')
for word, pos in prochesmots_maladie.iterrows():
    ax.annotate(word, pos, color = 'purple')
for word, pos in prochesmots_jaune.iterrows():
    ax.annotate(word, pos, color = 'pink')
for word, pos in prochesmots_mots_cibles.iterrows():
    ax.annotate(word, pos, color = 'blue')

plt.title('Modèle Word2Vec : Skip Gram : CORPUS PRESSE')
plt.grid()
plt.legend()

plt.savefig('model_sg_press.png')
plt.show()


##################################################################################################################
#                                           fasttext : CBOW
##################################################################################################################

# ouvrir le modèle et chercher les mots les plus proches : 
model_ft_press = FastText.load("fasttext_CBOW_press.model")


# chercher les 10 plus proches mots sans les scores de similarité de 4 mots 
prochesmots_patients_list,prochesmots_traitement_list,prochesmots_maladie_list,prochesmots_jaune_list = dix_mots_sans_score(model_ft_press)

data_press_liste = model_ft_press.wv.key_to_index.keys() # liste de tous les mots du corpus press 
#df = coord_2D(model_ft_press, data_press_liste) # coordonnees en 2D de tous les mots du corpus press 

# Pour enregistrer le file la première fois histoire car il prend du temps, puis on aura juste besoin de le charger.
#with open('results_TSNE_ft_press.pkl', 'wb') as file:    
    # A new file will be created
#    pickle.dump(df, file)

### Pour ouvrir le file une fois enregistré  ###
with open('results_TSNE_ft_press.pkl', 'rb') as file:     
    # Call load method to deserialze
    df = pickle.load(file)

# coordonnees des proches mots et des mots cibles 
prochesmots_mots_cibles = df.loc[mots_cibles]
prochesmots_patients = df.loc[prochesmots_patients_list] #patient 
prochesmots_traitement = df.loc[prochesmots_traitement_list] #traitement
prochesmots_maladie = df.loc[prochesmots_maladie_list] #maladie
prochesmots_jaune = df.loc[prochesmots_jaune_list] #jaune

# tracer le graphique 
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(prochesmots_patients['x'], prochesmots_patients['y'], color= 'red', label = 'PM patient')
ax.scatter(prochesmots_traitement['x'], prochesmots_traitement['y'], color= 'green',label = 'PM traitement')
ax.scatter(prochesmots_maladie['x'], prochesmots_maladie['y'], color= 'purple', label = 'PM maladie')
ax.scatter(prochesmots_jaune['x'], prochesmots_jaune['y'], color= 'pink', label = 'PM jaune')
ax.scatter(prochesmots_mots_cibles['x'], prochesmots_mots_cibles['y'], color= 'blue', label = 'mots cibles')

for word, pos in prochesmots_patients.iterrows():
    ax.annotate(word, pos, color = 'red')
for word, pos in prochesmots_traitement.iterrows():
    ax.annotate(word, pos, color = 'green')
for word, pos in prochesmots_maladie.iterrows():
    ax.annotate(word, pos, color = 'purple')
for word, pos in prochesmots_jaune.iterrows():
    ax.annotate(word, pos, color = 'pink')
for word, pos in prochesmots_mots_cibles.iterrows():
    ax.annotate(word, pos, color = 'blue')

plt.title('Modèle FastText : CBOW : CORPUS PRESSE')
plt.grid()
plt.legend()

plt.savefig('model_ft_press.png')
plt.show()


