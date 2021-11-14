import tensorflow.keras.backend as K
import cv2
import numpy as np
import tensorflow as tf


def valacc(model,count):

    wlen = 7

    labels_to_names = {0:u'..', 1:u'(', 1:u',', 1:u'<', 1:u'@', 1:u'\\',1:u'`',1:u'|',1:u'#',1:u'\'',1:u'+',1:u'/',1:u';',1:u'?',1:u'[',1:u'_',1:u'{',1:u'\"', 1:u'&',1:u'*',1:u'.',1:u':',1:u'>',1:u'^',1:u'~',1:u'!',1:u'%',1:u')',1:u'-',1:u'=',1:u'}',2:u'0',3:u'1',4:u'2',5:u'3',6:u'4',7:u'5', 8:u'6',9:u'7',10:u'8',11:u'9',12:u'a',12:u'A',13:u'b',13:u'B',14:u'c',14:u'C',15:u'd',15:u'D',16:u'e',16:u'E',17:u'f',17:u'F',18:u'g',18:u'G',19:u'h',19:u'H',20:u'i',20:u'I',21:u'j',21:u'J',22:u'k',22:u'K',23:u'l',23:u'L',24:u'm',24:u'M',25:u'n',25:u'N',26:u'o',26:u'O',27:u'p',27:u'P',28:u'q',28:u'Q',29:u'r',29:u'R', 30:u's',30:u'S',31:u't',31:u'T',32:u'u',32:u'U',33:u'v',33:u'V',34:u'w',34:u'W',35:u'x',35:u'X',36:u'y',36:u'Y',37:u'z',37:u'Z',1:u']',1:u'$'}

    acc_seq = 0
    acc_seg = 0

    for i in range(1,count):
        img = cv2.imread('./data/Val/imgs/'+str(i)+'.png')
        p = open('./data/Val/words/'+str(i)+'.txt')
        w = p.readlines()
        word = w[0]


        img = tf.keras.applications.resnet50.preprocess_input(img)
        img = cv2.resize(img,(256,64))
        nimg = np.expand_dims(img,axis=0)
        msk, wrd = model.predict(nimg)

        msk = np.argmax(msk,axis=-1)
        msk = np.squeeze(msk)
        msk = np.array(msk,dtype=np.uint8)

        wrd = np.argmax(wrd,axis=-1)

        wr = ''
        for w in wrd[0]:
            if w==0:
                break
            elif w==1:
                continue
            else:
                wr += labels_to_names[w]

        if wr.lower()==word.lower():
            acc_seq+=1
        print(word)
        print(wr)

        msk[msk==38]=0
        contours, hierarchy = cv2.findContours(msk,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        bb = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            bb.append([x,y,x+w,y+h])
    
        bb = np.array(bb)
        idx = np.argsort(bb,axis=0)
        if len(bb)!=0:
            bb = bb[idx[:,0]]

            wr1 = ''
            for c in bb:
                w = msk[c[1]:c[3],c[0]:c[2]]
                w = np.ravel(w)
                w = w[w!=38]
                counts = np.bincount(w)
                ch = np.argmax(counts)
                if ch!=1 and ch!=38:
                    ch = labels_to_names[ch]
                    wr1 += ch
                else:
                    continue

            if wr1.lower()==word.lower():
                acc_seg+=1

    print('Sequence Accuracy: ', acc_seq/count)
    print('SemSeg Accuracy: ', acc_seg/count)

    return acc_seq/count, acc_seg/count
