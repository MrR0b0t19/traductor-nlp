# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:52:23 2023

@author: Fantasma
"""
#importamos las librerias

import numpy as np 
import math 
import re
import time 
import tensorflow as tf 
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds 


#caragr datos
#nombre de el fichero de datos, en moodo lectura y el tipo de encodeado todo eso se llamara f y leemos eso 
with open("datos", 
          mode = "r", encoding = "uft-8") as f:
    datos= f.read()
    
    
with open("datos_cl", 
          mode = "r", encoding = "uft-8") as f:
    datos_cl= f.read()
    
    #dependiendo a lo que sea tiene que ir en corpus fragmentos de with open por cada lectura de dataset o corpus
    
#limpieza de texto 

datos_cl = datos_cl.split("\n")
datos_cl = [' ' + pref + '.' for pref in datos_cl]

corpus = datos
#añadimos $$$ despues delos puntos de frases sin fin 

for prefix in datos_cl:
    corpus =corpus.replace(prefix, prefix + '$$$')
corpus = re.sub(r"\.(?=[0-9] | [a-z] | [A-Z]", ".$$$", corpus) #re sub para reemplazar i substituir 
##realizo la eliminacion de marcadores $$$ 

corpus = re.sub(r"\.\$\$\$", '', corpus)

###eliminamos espacios multiples
corpus = re.sub(r"  +", " ", corpus)
corpus = corpus.split('\n')
#### se repuite los pasos si es que tienes mas corpus 


####tokenizar texto 

tokenizar = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    corpus, target_vocab_size = 2**13)
##utilizamos la libreria tfds con los demas segmentos para tokenizar el texto 


VOCAB_SIZE = tokenizar.vocab_size + 2 ### sumamois dos ceros para los valores grandes y pequeños 


inputs = [[VOCAB_SIZE-2] + tokenizar.encode(sentence) + [VOCAB_SIZE-1] 
          for sentence in corpus]

### lo mismo para hacefr elo outputs 
###outputs = [[]]

###ELMINAMOS LAS FRASES DEMASIADO LARGAS

MAX_LENGHT = 20

idx_to_remove = [count for count, sent in enumerate(inputs)
                 if len(sent) > MAX_LENGHT]
#for idx in reserved (idx_to_remove):
    #del inputs[idx]
   # del outputs[idx]
    ###repites por inputs o outputs
##idx_to_remove = [count for count, sent in enumerate(outputs)]
    
####CREAMOS LA ENTRADAS Y LAS SALIDAS 


inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                       value = 0,
                                                       padding = 'post',
                                                       maxlen = MAX_LENGHT)

###losmismo con el outputs

###CREAMOS LA ENTRADAS Y SALIDAS

## con un batch size y un bnuffer para los datasets 

# BATCH_SIZE = 64 
# BUFFER_SIZE = 20000

#dataset = tf.data.Dataset.from_tensor_slices((inputs##, outputs))

##dataset = dataset.cache()
#dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#dataset = dataset.shuffle(tf.data.experimental.AUTOTUNE)

###CREAREMOS UNA LAYER

class Pencoder(layers.Layer):### embedding para el encoding posicional para heredar de layers para crear nuestro propio layer
    
    def __init__(self):# aopartir de una clase se crea el objeto y se define primero __init__ 
        super(Pencoder, self).__init__()#llamara ala clase
        
    def get_angles(self, pos, i, d_model):##crear metodo pos = posicion de palabras, i= dimencion que estamos midiendo, d_model=dimencion total
        angles = 1 /np.power(10000., (2*(i//2))/ np.float32(d_model)) ##se usa power para elevar la potencia  y se pasa a notacion matematica
        return pos * angles##se hahran las multiplicaciones matriciales 
    
    
    def call(self, inputs): ##llamamos al metodo que se utilizara 
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]##accedemos alas dimenciones del modelo 
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],#el primer parametro van todos  al segundo eje para crear nueva dimensionn 
                                 np.arange(d_model)[np.newaxis, :],#
                                 d_model)#tomando secuencias de python
        angles[:, 0::2] = np.sin(angles[:, 0::2])#secuencias de listas 
        angles[:, 0::1] = np.cos(angles[:, 1::2])#para entrar alas pares 
        pos_enco = angles[np.newaxis, ...] ##
        return inputs + tf.cast(pos_enco, tf.float32)##lo volvemos untensor de 32 bits para combinar con el siguiente bloqe
    
##BLOQUE DE CALCULO DE ATENCION 
        
def sc_atten(queries, keys, values, mask): 
    product = tf.matmul(queries, keys, transpose_b = True) #matmul para multiplicar matrices 
    
    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)#casteamos a 32 bits
    sc_pro = product / tf.math.sqrt(keys_dim)##dividimos el dk 
    
    ##creamos la mascara para que no tome datos que no le hemos dado aun 
    if mask is not None:
        sc_pro += (mask * -1e9) #hara multiplicaacion a 0 * infiinito y luego si existe algo nos dara 1 y entonces nop 
        
    att = tf.matmul(tf.nn.softmax(sc_pro, axis = -1), values) #clculamos la atencion con softmax lo ponemos a cada uno de nuestros ejes 
    #estara nuestro producto escalar escalado 
    return att


##HACER UN MULTI ATENCVION POR CAPAS
class multheadatt(layers.layer):
    
    def __init__(self,nb_proj):
        super(multheadatt, self).__init__()
        self.nb_proj = nb_proj
        
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.nb_proj == 0

        self.d_proj = self.d_model // self.nb_proj
        
        self.query_lin = layers.Dense(units=self.d_model)
        self.key_lin = layers.Dense(units=self.d_model)
        self.value_lin = layers.Dense(units= self.d_model)
        
        self.final_lin = layers.Dense(units=self.d_model)
        
    
    def split_proj(self, inputs, batch_size):
        shape = (batch_size,
                 -1,
                 self.nb_proj,
                 self.d_proj)
        spl_inp = tf.reshape(inputs, shape= shape)
        return tf.transpose(spl_inp, perm=[0,2,1,3])
    
    def call (self, queries, keys, values, mask): #creamnos la mask y las agrupaciones creamos los bloques generales
        batch_size = tf.shape(queries)[0] #tamaño de batch con la primera dimencion [0]
        
        #cremoas transformnaciones lineales 
        queries = self.key_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)
        #tres capas densas  con esos datos para dividirlos
        
        
        
        queries = self.split_proj(queries, batch_size)
        keys = self.split_proj(keys, batch_size)
        values = self.split_proj(values, batch_size)
        
        att = sc_atten(queries, keys, values, mask)
        
        att = tf.transpose(att, perm=[0, 2, 1, 3])
        
        concat_att = tf.reshape(att,
                                      shape = (batch_size, -1, self.d_model))
        
        outputs = self.final_lin(concat_att)
        
        return outputs
    
        
    
    

