import nltk 
from nltk import word_tokenize
nltk.download("punkt")
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import pandas as pd
import keras.utils
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
import tqdm
import pickle

SEQUENCE_LENGTH = 100 # Länge der trainierten Sequence (Anzahl Wörter Pro Beispiel)
EMBEDDING_SIZE = 100  # Dimension des verwendeten Embeddings
BATCH_SIZE = 128 # Größe des trainierten Batches
EPOCHS = 10 # Anzahl der trianierten Epochen
MAX_FEATURES = 5000



dfs = []
#Einlesen der Plenarprotokolle 
for i in range(165,175):
    protokoll_path = f"../Data Scince App Hausarbeit 2/TXT/{i}.csv"
    dfs.append(pd.read_csv(protokoll_path, encoding = 'unicode_escape', sep = ",", error_bad_lines = False))
    
#Zusammenfügen der Einzelnen Plenarprotokolle als ein Pandas Dataframe
speeches_concat = pd.concat(dfs, ignore_index = True)

#Filtern der Daten, sodass nur notwendige Daten erhalten bleiben 
filtered_column_speeches = speeches_concat[["speaker_party","text","top","type"]]
filtered_row_speeches = filtered_column_speeches[
                                                 (filtered_column_speeches["speaker_party"].notnull()) &
                                                 (filtered_column_speeches["type"] == "speech") &
                                                 (filtered_column_speeches["speaker_party"] == "cducsu")]


#Alle Reden als zusammenhängenden Text darstellen
all_speeches_series = pd.Series(list(filtered_row_speeches["text"]))
all_speeches_text = all_speeches_series.str.cat(sep=' ')

#Text in einezelne Wörter zerlegen 
tokens_speeches = word_tokenize(all_speeches_text)

#max_features = es werden sich nur die X Häufigsten Wörter angeschaut
cv = CountVectorizer(max_features= MAX_FEATURES, lowercase= False, token_pattern ="(.*)")
cv.fit(tokens_speeches)

#features = X Häufigsten Wörter
features = cv.get_feature_names()


word_to_int = {}
int_to_word = {}

#jedes wort wird einer zahl zugeordnet
for i in range(0, len(features)):
    word = features[i]
    word_to_int[word] = i
    int_to_word[i] = word 

#Alle tokens wird in eine zahl umgewandelt allerdings nur die die auch zu denn 5000 meisten gehören
tokens_transformed = [word_to_int[word] for word in tokens_speeches if word in word_to_int]


with open("word_to_int.speechGeneration1.pickle", "wb") as file:
  pickle.dump(word_to_int, file)

with open("int_to_word.speechGeneration1.pickle", "wb") as file:
  pickle.dump(int_to_word, file)

X = []
y = []
# x = vektor an zahlen (satz)
# y = nächstes wort (One-Hot-Encoding) 
for i in range(0, len(tokens_transformed) - SEQUENCE_LENGTH):
    X.append(tokens_transformed[i:i + SEQUENCE_LENGTH])
    y.append(tokens_transformed[i+SEQUENCE_LENGTH])
X = np.array(X)
y = np.array(to_categorical(y, num_classes = cv.max_features))





def get_embedding_vectors(dim=100):
    embedding_index = {}
    #Einlesen Pretrained Embedding
    with open(f"model.txt", encoding='unicode_escape') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
  
            values = line.split()
            if not values:
              break
            try:
                #Erster Wert im Array ist immer das Wort
                word = values[0]
                #Restlichen Werte repräsentieren den Vektor
                vectors = np.asarray(values[1:], dtype='float64')
                embedding_index[word] = vectors
            except ValueError:            
                pass
    word_index = word_to_int
    embedding_matrix = np.zeros((len(word_index), dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix




embedding_matrix = get_embedding_vectors()
#Input_Layer
input1 = layers.Input((SEQUENCE_LENGTH))
input2 = layers.Input((SEQUENCE_LENGTH))

#Embedding_Layer
pretrained_Embedding = layers.Embedding(cv.max_features,
                       EMBEDDING_SIZE,
                       weights=[embedding_matrix],
                       trainable=False,
                       input_length=SEQUENCE_LENGTH)(input1)

selftrained_Embedding = layers.Embedding(cv.max_features, 
                        EMBEDDING_SIZE,
                        trainable=True,
                        input_length=SEQUENCE_LENGTH)(input2)

#Embedding_Layer zusammenfügen
layerlist = [pretrained_Embedding, selftrained_Embedding]
concat = layers.Concatenate(axis = -1)(layerlist)
#LSTM first Hiddenlayer
first_LSTM = layers.LSTM(256, return_sequences = True)(concat)
#Dropout (Präventativ gegen mögliches overfitting)
dropout_015 = layers.Dropout(0.15)(first_LSTM)
#LSTM second Hiddenlayer
second_LSTM = layers.LSTM(256, return_sequences = False)(first_LSTM)
#Dense third Hiddenlayer
first_dense = layers.Dense(10000,activation= "sigmoid")(second_LSTM)
#Dense Output_Layer
output = layers.Dense(cv.max_features, activation="softmax")(second_LSTM)

#Model Trainieren 
model = models.Model([input1,input2],output)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
              metrics=["accuracy",tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


# In[24]:


# initialize our ModelCheckpoint and TensorBoard callbacks
# model checkpoint for saving best weights
model.fit([X,X],y, batch_size=BATCH_SIZE, epochs=EPOCHS,verbose=1)
model.save("speechGeneration1.pickle")

