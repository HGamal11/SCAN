import cv2
import numpy as np
import tensorflow as tf
import random
import os
import keras

def preprocess_word(x):

    names_to_labels = {u'..':0, u'(':1, u',':1, u'<':1, u'@':1, u'\\':1,u'`':1,u'|':1,u'#':1,u'\'':1,u'+':1,u'/':1,u';':1,u'?':1,u'[':1,u'_':1,u'{':1,u'\"':1, u'&':1,u'*':1,u'.':1,u':':1,u'>':1,u'^':1,u'~':1,u'!':1,u'%':1,u')':1,u'-':1,u'=':1,u'}':1,u'0':2,u'1':3,u'2':4,u'3':5,u'4':6,u'5':7,u'6':8,u'7':9,u'8':10,u'9':11,u'a':12,u'A':12,u'b':13,u'B':13,u'c':14,u'C':14,u'd':15,u'D':15,u'e':16,u'E':16,u'f':17,u'F':17,u'g':18,u'G':18,u'h':19,u'H':19,u'i':20,u'I':20,u'j':21,u'J':21,u'k':22,u'K':22,u'l':23,u'L':23,u'm':24,u'M':24,u'n':25,u'N':25,u'o':26,u'O':26,u'p':27,u'P':27,u'q':28,u'Q':28,u'r':29,u'R':29,u's':30,u'S':30,u't':31,u'T':31,u'u':32,u'U':32,u'v':33,u'V':33,u'w':34,u'W':34,u'x':35,u'X':35,u'y':36,u'Y':36,u'z':37,u'Z':37,u']':1,u'$':1}
    w = np.zeros(32)
    m = 0   
    for c in x:
        try:
            w[m] = names_to_labels[c]
        except:
            continue
        m+=1

    w_onehot = np.zeros((32, 38), dtype='float32')
    t = 0
    for r in w:
        w_onehot[t, int(r)] = 1.
        t += 1
    return w

def preprocess_mask(x):
    batch_size, n_rows, n_cols = x.shape[:3]
    x1d = x.ravel()
    n_cls = 38
    y1d = tf.keras.utils.to_categorical(x1d, num_classes = n_cls)
    y4d = y1d.reshape([batch_size, n_rows, n_cols, n_cls])
    return y4d

def Generator_stg1(batch_size):

    DIR = './data/imgs/'
    no_imgs = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    r = list(range(1,no_imgs))
    random.shuffle(r)
    idx = 0
    c = 0
    while True:
        x  = []
        y = []
        y1 = []

        for m in range(batch_size):
            try:
                i = r[idx]
            except:
                idx = 0
                r = list(range(1,no_imgs))
                random.shuffle(r)
            idx+=1

            img = cv2.imread('./data/imgs/'+str(i)+'.png')
            img = tf.keras.applications.resnet50.preprocess_input(img)
            img = cv2.resize(img,(256,64))

            msk = cv2.imread('./data/masks/'+str(i)+'.png',0)
            msk = cv2.resize(msk,(256,64),interpolation =cv2.INTER_NEAREST)

            wrd = open('./data/words/'+str(i)+'.txt')
            w = wrd.readlines()

            try:
                wrd = w[0]
            except:
                batch_size+=1
                continue

            y_wrd = preprocess_word(wrd)
            y_wrd = np.expand_dims(y_wrd,axis=0)
            y1.append(y_wrd)
            x.append(img)
            y.append(msk)

        x = np.array(x)
        y = np.array(y)#/38.0
        y1 = np.array(y1)
        yield [x,y,y1],[]



def Generator_stg2(batch_size):

    #synthetic data

    DIR = './data/imgs/'
    no_imgs = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    r = list(range(0, no_imgs))
    random.shuffle(r)

    # real data
    grt = []
    gt = open('./Real.txt')  #Real.txt contains the info (path,word) for all the real data used for training from different datasets
    for line in gt:
        grt.append(line)
    r1 = list(range(1, len(grt)))
    random.shuffle(r1)

    chng = 0
    idx = 0
    idx1 = 0
    while True:
        x = []
        y = []

        for m in range(batch_size):

            if chng == 0:
                try:
                    i = r[idx]
                except:
                    r = list(range(0, no_imgs))
                    random.shuffle(r)
                    idx = 0
                idx += 1

                img = cv2.imread('./data/imgs/' + str(i) + '.png')
                img = tf.keras.applications.resnet50.preprocess_input(img)
                img = cv2.resize(img, (256, 64))

                wrd = open('./data/words/' + str(i) + '.txt')
                w = wrd.readlines()

            else:
                try:
                    i1 = r1[idx1]
                except:
                    r1 = list(range(1, len(grt)))
                    random.shuffle(r1)
                    idx1 = 0
                idx1 += 1

                line = grt[i1]
                line = line.split(',')
                path = line[0]
                path = path.replace('\ufeff', '')
                word = line[1]
                word = word.replace(' ', '')
                word = word.replace('\n', '')
                word = word.replace('"', '')
                img = cv2.imread(path)

                img = keras.applications.resnet50.preprocess_input(img)
                img = cv2.resize(img, (256, 64))
                wrd = str(word)

            chng = random.randint(0,5)

            y_wrd = preprocess_word(wrd)

            x.append(img)
            y.append(y_wrd)

        x = np.array(x)
        y = np.array(y)

        yield x, y
