import tensorflow_datasets as tfds
import tensorflow as tf


dataset = tfds.load('imagenet2012_subset', split='train', download=True)
dataset = dataset.shuffle(1024)
dataset = dataset.batch(1)
x = next(iter(dataset))
print(x)
img   = x['image']
label = x['label']
VGG = tf.keras.applications.vgg16.VGG16()
img = tf.image.resize(img, (224, 224))
img = tf.keras.applications.vgg16.preprocess_input(img)
y   = VGG(img)
print(label[0])
print(y[:,label[0]])
#print(y)
print(tf.math.argmax(y[0]))
print(tf.reduce_max(y[0]))
print(*sorted(VGG.__dir__()), sep='\n')
#print(tf.keras.applications.vgg16.decode_predictions(y))