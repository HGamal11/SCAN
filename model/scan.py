
import keras
from keras_self_attention import SeqSelfAttention
from utils.CustomMultiLossLayer import CustomMultiLossLayer
from model.hrnet import seg_hrnet

def create_word_model(out_seg, max_seq_length, num_classes):

    out = keras.layers.Permute((2,1,3))(out_seg)

    out = keras.layers.Conv2D(64,kernel_size=(3,3),strides=(1,2),activation='relu',padding='same')(out)
    out = keras.layers.Conv2D(64,kernel_size=(3,3),strides=(2,2),padding='same')(out)
    out = keras.layers.Dropout(.3)(out)
    out = keras.layers.Activation('relu')(out)

    out = keras.layers.Conv2D(128,kernel_size=(3,3),strides=(1,2),activation='relu',padding='same')(out)
    out = keras.layers.Conv2D(128,kernel_size=(3,3),strides=(2,2),padding='same')(out)
    out = keras.layers.Dropout(.3)(out)
    out = keras.layers.Activation('relu')(out)

    out = keras.layers.Conv2D(256,kernel_size=(3,3),strides=(1,2),padding='same')(out)  #bef:(2,2)
    out = keras.layers.Dropout(.3)(out)
    out = keras.layers.Activation('relu')(out)

    out = keras.layers.Conv2D(512,kernel_size=(3,3),strides=(2,2),padding='same')(out) #(2,1)  1024:(2,2)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('relu')(out)
    
    out = keras.layers.Reshape((max_seq_length,512))(out)
    out = keras.layers.Dense(256,activation='relu')(out)

    out = keras.layers.Bidirectional(keras.layers.LSTM(512,return_sequences=True))(out)

    SA_l = SeqSelfAttention(attention_activation='sigmoid')
    out = SA_l(out)
    out = keras.layers.LSTM(512,return_sequences=True)(out)
    out = keras.layers.TimeDistributed(keras.layers.Dense(num_classes, activation='softmax',name='out11'),name='out1')(out)
    
    return out


def scan(img_size,num_classes,max_seq_length,mode):

    img_w, img_h, channels= img_size[0],img_size[1], img_size[2]

    if keras.backend.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(channels, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, channels))

    out_seg = seg_hrnet(inputs,img_w,img_h, channels, num_classes)
    
    y_true_seg = keras.layers.Input(shape=(img_w,img_h))    #segmentation groundtruth
    y_true_word = keras.layers.Input(shape=(None,max_seq_length))  #word groundtruth

    out_word = create_word_model(out_seg, max_seq_length, num_classes)

    outloss = CustomMultiLossLayer(nb_outputs=2)([y_true_seg, y_true_word, out_seg, out_word])

    if mode=='stg1':
        pred_model = keras.models.Model(inputs=inputs, outputs=[out_seg,out_word])
        train_model = keras.models.Model(inputs=[inputs, y_true_seg, y_true_word],outputs=outloss)
    else:
        train_model = keras.models.Model(inputs=inputs, outputs=[out_word])
        pred_model = keras.models.Model(inputs=inputs, outputs=[out_seg,out_word])
    
    return train_model, pred_model
    
