import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Brining in tensorflow datasets for fashion mnist
import tensorflow_datasets as tfds
# Bringing in matplotlib for viz stuff
from matplotlib import pyplot as plt
# Use the tensorflow datasets api to bring in the data source
ds = tfds.load('fashion_mnist', split='train')

# Do some data transformation
# Setup connection aka iterator
dataiterator = ds.as_numpy_iterator()
# Setup the subplot formatting
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 20))
# Loop four times and get images
for idx in range(4):
    for jdx in range(4):
        sample = dataiterator.next()
        ax[idx][jdx].imshow(np.squeeze(sample['image']))
        ax[idx][jdx].title.set_text(sample['label'])
# plt.show()

# Scale and return images only
def scale_images(data):
    image = data['image']/255
    return image

# Running the dataset through the scale_images preprocessing step
ds = ds.map(scale_images)
#for i in ds:
#    print(i)

# Shuffle it up
ds = ds.shuffle(60000)
# Batch into 128 images per sample
ds = ds.batch(128)
# Cache the dataset for that batch
ds = ds.cache()
# Reduces the likelihood of bottle necking
ds = ds.prefetch(tf.data.AUTOTUNE)

# Bring in the sequential api for the generator and discriminator
from keras.models import Sequential
# Bring in the layers for the neural network
from keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D

def build_generator():
    model = Sequential()

    # Takes in random values and reshapes it to 7x7x128
    # Beginnings of a generated image
    model.add(Dense(7*7*128, input_dim=128))
    model.add(Reshape((7, 7, 128)))

    # Up-sampling block 1
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same', activation='relu'))

    # Up-sampling block 2
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same', activation='relu'))

    # Convolutional block 1
    model.add(Conv2D(128, 3, padding='same', activation='relu'))

    # Convolutional block 2
    model.add(Conv2D(128, 3, padding='same', activation='relu'))

    # Conv layer to get to one channel
    model.add(Conv2D(1, 3, padding='same', activation='sigmoid'))

    return model


#Test generator
generator = build_generator()
#generator.summary()

# Generate new fashion
img = generator.predict(np.random.randn(5, 128, 1))
# Setup the subplot formatting
fig, ax = plt.subplots(ncols=5, figsize=(28, 28))
# Loop four times and get images
for idx, img in enumerate(img):
    # Plot the image using a specific subplot
    ax[idx].imshow(np.squeeze(img))
    # Appending the image label as the plot title
    ax[idx].title.set_text(idx)
#plt.show()

def build_discriminator():
    model = Sequential()
    # First Conv Block
    model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    # Second Conv Block
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    # Third Conv Block
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.6))
    # Fourth Conv Block
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.6))
    # Flatten then pass to dense layer
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model

discriminator = build_discriminator()
#discriminator.summary()

# Adam is going to be the optimizer for both
from keras.optimizers import Adam
# Binary cross entropy is going to be the loss for both
from keras.losses import BinaryCrossentropy


# Importing the base model class to subclass our training step
from keras.models import Model
class FashionGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        # Pass through args and kwargs to base class
        super().__init__(*args, **kwargs)

        # Create attributes for gen and disc
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, *args, **kwargs):
        # Compile with base class
        super().compile(*args, **kwargs)

        # Create attributes for losses and optimizers
        self.g_opt = Adam(learning_rate=0.001)
        self.d_opt = Adam(learning_rate=0.0001)
        self.g_loss = BinaryCrossentropy()
        self.d_loss = BinaryCrossentropy()

    def train_step(self, batch):
        # Get the data
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)
        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create labels for real and fakes images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Calculate loss - BINARYCROSS
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)
        # Apply backpropagation - nn learn
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate some new images
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)

            # Create the predicted labels
            predicted_labels = self.discriminator(gen_images, training=False)

            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)
        # Apply back-propagation
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss": total_d_loss, "g_loss": total_g_loss}


# Create instance of subclassed model
fashgan = FashionGAN(generator, discriminator)
# Compile the model
fashgan.compile()

# Recommend 2000 epochs
total_loss = fashgan.fit(ds, epochs=20)

generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')

