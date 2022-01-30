
# Importer les packages qu'il faut 
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
#import fasttext
import re 

# récupérer les 2 chemins de mes dossiers 
path_med = "/Users/chaimae/Desktop/AgroParisTech/3A/cours_sup_IA/Chatbot_Text_Mining/TP1/TP_ISD2020/QUAERO_FrenchMed/QUAERO_FrenchMed_traindev.ospl"
path_press = "/Users/chaimae/Desktop/AgroParisTech/3A/cours_sup_IA/Chatbot_Text_Mining/TP1/TP_ISD2020/QUAERO_FrenchPress/QUAERO_FrenchPress_traindev.ospl"

# découper les mots en liste de vecteurs. 

def load_data(path):
    a_file = open(path,"r")
    list_of_lists = []
    for line in a_file :
      # enlever la ponctuation pour ne pas avoir parmi les 10 plus proches mots de la ponctuation. 
      stripped_line = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]','',line).strip()
      line_list = stripped_line.split() # diviser le mot
      line_list = [x for x in line_list if len(x)>1] # enlever les mots inférieur à taille 1
      list_of_lists.append(line_list)
    
    a_file.close()
    return list_of_lists


# Les données : des listes de listes sans ponctuation ni mots de taille inférieur à 1
data_med = load_data(path_med)
data_press = load_data(path_press)

# Un fichier .txt où je souhaite enregistrer les mots les plus proches par la suite pour les avoir à porté de main
file = open("/Users/chaimae/Desktop/AgroParisTech/3A/cours_sup_IA/Chatbot_Text_Mining/TP1/WordEmbeddings_Results.txt", "w") 
file.write(" WORD EMBEDDINGS : 10 MOST SIMILAR WORDS IN A MEDICAL AND PRESS CORPUS  \n \n \n \n  ")


###############################################################################################################
#                                        Word2VC : CBOW 
###############################################################################################################

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'                                            Medical                                                       ' 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# initialiser le modèle 
model_CBOW_med = Word2Vec(sentences=data_med, vector_size=100, window=5, min_count=1, workers=4)

#Entrainer le modèle 
model_CBOW_med.train(data_med, total_examples=1, epochs=50)


# récupérer la liste des 10 mots les plus proches et leur score de similarité
sims_patient_CBOW_med = model_CBOW_med.wv.most_similar('patient', topn=10)  # récupérer 10 mots similaires
sims_traitement_CBOW_med = model_CBOW_med.wv.most_similar('traitement', topn=10)  # récupérer d'autre mots similaires
sims_maladie_CBOW_med= model_CBOW_med.wv.most_similar('maladie', topn=10)  # récupérer d'autre mots similaires
sims_jaune_CBOW_med= model_CBOW_med.wv.most_similar('jaune', topn=10)  # récupérer d'autre mots similaires

#enregistrer le modèle
model_CBOW_med.save("word2vec_CBOW_med.model")

# Enregistrer les mots + leurs embeddings
#word_vectors = model_CBOW_med.wv
#word_vectors.save("word2vec_CBOW_med.wordvectors")

# écrire les 10 mots les plus proches dans le fichier .txt
file.write("------------------- Word2VC : CBOW ---------------------------------------------------- \n \n \n \n ")
file.write("######## Medical corpus ############ \n \n \n ")
file.write("Word PATIENT : \n %s \n \n Word TRAITEMENT : \n %s \n \n  Word MALADIE : \n %s \n \n Word JAUNE : \n %s \n \n \n "%(sims_patient_CBOW_med,sims_traitement_CBOW_med,sims_maladie_CBOW_med, sims_jaune_CBOW_med))

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'                                            Press                                                         ' 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# initialiser le modèle 
model_CBOW_press = Word2Vec(sentences=data_press, vector_size=100, window=5, min_count=1, workers=4)

#entraîner le modèle
model_CBOW_press.train(data_press, total_examples=1, epochs=50)

# récupérer la liste des 10 mots les plus proches et leur score de similarité 
sims_patient_CBOW_press = model_CBOW_press.wv.most_similar('patient', topn=10)  # récupérer d'autre mots similaires
sims_traitement_CBOW_press = model_CBOW_press.wv.most_similar('traitement', topn=10)  # récupérer d'autre mots similaires
sims_maladie_CBOW_press= model_CBOW_press.wv.most_similar('maladie', topn=10)  # récupérer d'autre mots similaires
sims_jaune_CBOW_press= model_CBOW_press.wv.most_similar('jaune', topn=10)  # récupérer d'autre mots similaires

#enregistrer le modèle
model_CBOW_press.save("word2vec_CBOW_press.model")

# Stocker kes mots + leurs embeddings
#word_vectors = model_CBOW_press.wv
#word_vectors.save("word2vec_CBOW_press.wordvectors")

# écrire les 10 mots les plus proches dans le fichier .txt
file.write("######## Press corpus ############ \n \n \n")
file.write("Word PATIENT : \n %s \n \n Word TRAITEMENT : \n %s \n \n Word MALADIE : \n %s \n \n Word JAUNE : \n %s \n \n \n"%(sims_patient_CBOW_press,sims_traitement_CBOW_press,sims_maladie_CBOW_press, sims_jaune_CBOW_press))


###############################################################################################################
#                                        Word2VC : Skip Gram
###############################################################################################################

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'                                            Medical                                                       ' 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# initialiser le modèle 
model_sg_med = Word2Vec(sentences=data_med, vector_size=100, window=5, min_count=1, workers=4, sg=1)

#entrainer le modèle 
model_sg_med.train(data_med, total_examples=1, epochs=50)

# récupérer la liste des 10 mots les plus proches et leur score de similarité
sims_patient_sg_med = model_sg_med.wv.most_similar('patient', topn=10)  # récupérer d'autre mots similaires
sims_traitement_sg_med = model_sg_med.wv.most_similar('traitement', topn=10)  # récupérer d'autre mots similaires
sims_maladie_sg_med= model_sg_med.wv.most_similar('maladie', topn=10)  # récupérer d'autre mots similaires
sims_jaune_sg_med= model_sg_med.wv.most_similar('jaune', topn=10)  # récupérer d'autre mots similaires

#enregistrer le modèle
model_sg_med.save("word2vec_sg_med.model")

# Stocker kes mots + leurs embeddings
#word_vectors = model_sg_med.wv
#word_vectors.save("word2vec_sg_med.wordvectors")

# écrire les 10 mots les plus proches dans le fichier .txt
file.write("------------------- Word2VC : Skip Gram ---------------------------------------------------- \n \n \n \n")
file.write("######## Medical corpus ############ \n \n \n")
file.write("Word PATIENT : \n %s \n \n Word TRAITEMENT : \n %s \n \n  Word MALADIE : \n %s \n \n Word JAUNE : \n %s \n \n \n"%(sims_patient_sg_med,sims_traitement_sg_med,sims_maladie_sg_med, sims_jaune_sg_med))


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'                                            Press                                                         ' 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# initialiser le modèle 
model_sg_press = Word2Vec(sentences=data_press, vector_size=100, window=5, min_count=1, workers=4, sg =1)

# entrainer le modèle
model_sg_press.train(data_press, total_examples=1, epochs=50)

# récupérer la liste des 10 mots les plus proches et leur score de similarité
sims_patient_sg_press = model_sg_press.wv.most_similar('patient', topn=10)  # récupérer d'autre mots similaires
sims_traitement_sg_press = model_sg_press.wv.most_similar('traitement', topn=10)  # récupérer d'autre mots similaires
sims_maladie_sg_press= model_sg_press.wv.most_similar('maladie', topn=10)  # récupérer d'autre mots similaires
sims_jaune_sg_press= model_sg_press.wv.most_similar('jaune', topn=10)  # récupérer d'autre mots similaires

#enregistrer le modèle
model_sg_press.save("word2vec_sg_press.model")

# Stocker kes mots + leurs embeddings
#word_vectors = model_sg_press.wv
#word_vectors.save("word2vec_sg_press.wordvectors")

# écrire les 10 mots les plus proches dans le fichier .txt
file.write("######## Press corpus ############ \n \n \n")
file.write("Word PATIENT : \n %s \n \n Word TRAITEMENT : \n %s \n \n Word MALADIE : \n %s \n \n Word JAUNE : \n %s \n \n \n"%(sims_patient_sg_press,sims_traitement_sg_press,sims_maladie_sg_press, sims_jaune_sg_press))


###############################################################################################################
#                                            fasttex
###############################################################################################################

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'                                            Medical                                                       ' 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Entrainer le modèle
model_ft_med = FastText(data_med, vector_size=100 , min_count=1, sg = 0 , epochs=50)

# récupérer le vecteur du mot et les 10 plus proches mots 

sims_patient_ft_med = model_ft_med.wv.most_similar('patient', topn=10)
sims_traitement_ft_med = model_ft_med.wv.most_similar('traitement', topn=10)
sims_maladie_ft_med = model_ft_med.wv.most_similar('maladie', topn=10)
sims_jaune_ft_med = model_ft_med.wv.most_similar('jaune', topn=10)

#enregistrer le modèle
model_ft_med.save("fasttext_CBOW_med.model")
# Stocker kes mots + leurs embeddings
#word_vectors = model_ft_med.wv
#word_vectors.save("fasttext_CBOW_med.wordvectors")

# écrire les 10 mots les plus proches dans le fichier .txt
file.write("------------------- Fasttext : CBOW ---------------------------------------------------- \n \n \n \n ")
file.write("######## Medical corpus ############ \n \n \n ")
file.write("Word PATIENT : \n %s \n \n Word TRAITEMENT : \n %s \n \n  Word MALADIE : \n %s \n \n Word JAUNE : \n %s \n \n \n "%(sims_patient_ft_med,sims_traitement_ft_med,sims_maladie_ft_med, sims_jaune_ft_med))


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'                                            Press                                                         ' 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Entrainer le modèle
model_ft_press = FastText(data_press, vector_size=100 , min_count=1, sg = 0 , epochs=50)

# récupérer la liste des 10 mots les plus proches et leur score de similarité
sims_patient_ft_press = model_ft_press.wv.most_similar('patient', topn=10)
sims_traitement_ft_press = model_ft_press.wv.most_similar('traitement', topn=10)
sims_maladie_ft_press = model_ft_press.wv.most_similar('maladie', topn=10)
sims_jaune_ft_press = model_ft_press.wv.most_similar('jaune', topn=10)

#enregistrer le modèle
model_ft_press.save("fasttext_CBOW_press.model")

# Stocker kes mots + leurs embeddings
#word_vectors = model_ft_press.wv
#word_vectors.save("fasttext_CBOW_press.wordvectors")

# écrire les 10 mots les plus proches dans le fichier .txt
file.write("######## Press corpus ############ \n \n \n ")
file.write("Word PATIENT : \n %s \n \n Word TRAITEMENT : \n %s \n \n  Word MALADIE : \n %s \n \n Word JAUNE : \n %s \n \n \n "%(sims_patient_ft_press,sims_traitement_ft_press,sims_maladie_ft_press, sims_jaune_ft_press))


file.close()
