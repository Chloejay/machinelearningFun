import pandas as pd
import numpy as np 
import os
import keras
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard 
from keras.callbacks import EarlyStopping 
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 
%matplotlib inline  

os.environ['HOME']+'/.kaggle/datasets/zalando-research/fashionmnist'
train=pd.read_csv('./desktop/fashionmnist/fashion-mnist_train.csv') 
test= pd.read_csv('./desktop/fashionmnist/fashion-mnist_test.csv')

x_train_validate = train[list(train.columns)[1:]].\
    values/255 #to standardize the data for rescaling between (0,1] to remove stddev variance when calculate
#img processing is the huge part of the model, for this is all about pixel calculation with testing (w,b) with maths algorithm cost function, learning rate, optimizer function, SGD, by ConNets (image * filter) and make more dense of images to fine tunning of params 
y_train_validate = train['label'].values 
x_test= test[list(test.columns)[1:]].values/255 
y_test= test['label'].values 
#print(x_train_validate.shape, x_test.shape, y_train_validate.shape, y_test.shape)

from sklearn.model_selection import train_test_split
x_train, x_validate, y_train, y_validate= train_test_split(x_train_validate, 
                                                           y_train_validate, 
                                                           test_size=0.1, 
                                                           random_state=1234) 
#print(x_train.shape, x_validate.shape, y_train.shape, y_test.shape)

image= x_train[10000,:].reshape(28,28) 
plt.imshow(image) 
plt.show() 
print(x_train.shape) 

im_rows = 28
im_cols = 28
batch_size = 258
im_shape = (im_rows, im_cols, 1) 
x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)

cnn_model = Sequential([
    Conv2D(filters=32, 
           kernel_size=(3,3), 
           #the matrix that need to filter rows and columns of the image 
           activation='relu',
           #most valuable activation func to drop the irrelevant pixels
           input_shape=im_shape), 
    MaxPooling2D(pool_size=2),
    Dropout(0.2), 
    Flatten(), 
    Dense(32, activation='relu'), 
    Dense(10, activation='softmax') 
])

tensorboard = TensorBoard(
    log_dir='logs\{}'.format('cnn_1layer'),
    write_graph=True,
    write_grads=True,
    histogram_freq=10,
    write_images=True,
)

cnn_model.compile(
    optimizer=Adam(lr=0.001), #Adam and learning rate
    loss='sparse_categorical_crossentropy', #categorical cross entropy 
    metrics=['accuracy']
)

cnn_model.fit(
    x_train, 
    y_train, 
    batch_size=batch_size,
    epochs=10, 
    verbose=1,
    validation_data=(x_validate, y_validate),
    callbacks=[tensorboard]
)

score = cnn_model.evaluate(x_test, y_test, verbose=0)
print('test loss: {:.2f}'.format(score[0]))
print(' test acc: {:.2f}'.format(score[1]))

cnn_model.summary()

pred=cnn_model.predict_classes(x_test)  
print(pred)
print(pred.shape)

#History.history attribute is a record of training loss values and metrics values at successive epochs, 
#as well as validation loss values and validation metrics values 
History = cnn_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=20, 
    #the more epochs has more higher accuracy, I tried the epochs around 10, the accuracy is relatively not good
    #here need to understand how to tune the hyperparameters 
    verbose=1,
    validation_data=(x_validate,y_validate), 
)

#compare the training and validation datasets by the accuracy with the loss and metrics values with model we just trained 
accuracy = History.history['acc']
val_accuracy = History.history['val_acc']
loss = History.history['loss']
val_loss =History.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'co', label='Training Acc')
plt.plot(epochs, val_accuracy, 'orange', label='Validation Acc')
plt.title('Accuracy of Training and Validation dataset') 
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'c+', label='Training Loss')
plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
plt.title('Loss of Training and validation dataset')
plt.legend()
plt.show() 

y_pred = cnn_model.predict_classes(x_test) 
y_test = test[:, 0] 
correct = np.nonzero(pred==y_test)[0]
incorrect = np.nonzero(pred!=y_test)[0]

for i, correct in enumerate(correct[:9]): 
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28), 
               cmap='terrain', 
               interpolation= 'none')
    plt.title("pred {}, actual {}".format(y_pred[correct], y_test[correct]))
    plt.tight_layout()

embed_count = 800           
test=np.array(pd.read_csv('./desktop/fashionmnist/fashion-mnist_test.csv'),dtype='float32')   
x_test_tf = test[:embed_count, 1:] / 255
y_test_tf = test[:embed_count, 0]
print(x_test_tf)
print(y_test_tf)

#use tensorboard to turn the layer computation(black box) to flashlight layer, to visualize the graph 
#a python class to write data for tensorboard 
logdir='./desktop/fashionmnist/logdir'
writer = tf.summary.FileWriter(logdir) 
embedding_var = tf.Variable(x_test, name='fmnist_embedding')
config = projector.ProjectorConfig() 
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = logdir+'metadata.tsv'
embedding.sprite.image_path = logdir+ 'image.png' 
embedding.sprite.single_image_dim.extend([28, 28])

writer = tf.summary.FileWriter(logdir + 'visualization') #save the check point 

projector.visualize_embeddings(writer, config)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, logdir+'model.ckpt',0)  
    
writer.add_graph(sess.graph)

rows = 28
cols = 28
label = ['t_shirt', 'trouser', 
         'pullover', 'dress', 
         'coat','sandal', 
         'shirt', 'sneaker', 
         'bag', 'ankle_boot'] 
sprite_dim = int(np.sqrt(x_test.shape[0]))
sprite_image = np.ones((cols * sprite_dim, rows * sprite_dim))
index = 0
labels = []
for i in range(sprite_dim):
    for j in range(sprite_dim):
        labels.append(label[int(y_test[index])])
        sprite_image[
            i * cols: (i + 1) * cols,
            j * rows: (j + 1) * rows
        ] = x_test[index].reshape(28, 28) * -1 + 1
        index += 1 
        
with open(embedding.metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(labels):
        meta.write('{}\t{}\n'.format(index, label))
        
plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gist_rainbow')
plt.imshow(sprite_image, cmap='gist_rainbow')
plt.show()

#print('x_train shape: {}'.format(x_train.shape))
#print('x_test shape: {}'.format(x_test.shape))
#print('x_validate shape: {}'.format(x_validate.shape))





