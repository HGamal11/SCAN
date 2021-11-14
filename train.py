import keras
from model.scan import scan
from utils.dataGen import Generator_stg1,Generator_stg2
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.utils import multi_gpu_model
from utils.valTest import valacc


stg1= 1
stg2= 0
batch_size= 8
no_of_epochs= 5
steps_per_epoch= 50
num_classes= 38
img_size = (64,256,3)
max_seq_length = 32
no_of_val_imgs= 10

K.set_image_data_format('channels_last')

trmodel,pmodel = scan(img_size=img_size,max_seq_length=max_seq_length,num_classes=num_classes,mode='stg1')
trmodel.summary()

#**uncomment to load weights
#trmodel.load_weights('checkpoint/weights-improvement-scan.h5', by_name=True, skip_mismatch=True)

#**uncomment to use multigpu
#trmodel_m = multi_gpu_model(trmodel, gpus=2)

trmodel_m = trmodel

print('Model Loaded')

callbacks_list = []
reduce_lr = ReduceLROnPlateau(monitor='loss',mode = 'auto',cooldown=0 ,factor=0.8, patience=5, verbose=1, min_lr=1e-5)

class saveModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        filepath="./checkpoint/weights-improvement-scan.h5"
        trmodel.save_weights(filepath)
        acc,_ = valacc(pmodel,no_of_val_imgs)

checkpoint = saveModel()

callbacks_list.append(checkpoint)
callbacks_list.append(reduce_lr)

if stg1:
    trmodel_m.compile(optimizer=keras.optimizers.adam(1e-4,clipnorm=1.), loss = None, metrics=['accuracy'])
    train_generator = Generator_stg1(batch_size)
    trmodel_m.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=no_of_epochs,
        shuffle=False,
        callbacks=callbacks_list,
        use_multiprocessing=False)
else:
    trmodel_m.compile(optimizer=keras.optimizers.adam(1e-4,clipnorm=1.), loss ='categorical_crossentropy', metrics=['accuracy'])
    train_generator = Generator_stg2(batch_size)
    trmodel_m.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=no_of_epochs,
        shuffle=False,
        callbacks=callbacks_list,
        use_multiprocessing=False)



