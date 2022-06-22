import os

import numpy as np

# from autoencoder import VAE
from autoencoder_v2 import VAE_v2


LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150
# 1) STFT case
SPECTROGRAMS_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms"
TRAIN_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_train"
# 2) MCLT case
# SPECTROGRAMS_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_mclt"
# TRAIN_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_train_mclt"

def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path, allow_pickle=True) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths

def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE_v2(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        #latent_space_dim=128
        latent_space_dim=2
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train,_ = load_fsdd(TRAIN_PATH)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")