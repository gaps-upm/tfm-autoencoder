import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import os
import librosa
import soundfile as sf
import scipy.stats as st

from autoencoder import VAE
# from autoencoder_v2 import VAE_v2
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from train import load_fsdd
from soundgenerator import SoundGenerator

HOP_LENGTH = 256
SAMPLE_RATE = 22050
# 1) STFT case
TEST_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_test"
TRAIN_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_train"
LATENT_SPACE_SAVE_PATH = "/home/jaimelopez/TFM/datasets/fsdd/latent_space"
SAVE_DIR_GENERATED = "/home/jaimelopez/TFM/datasets/fsdd/STFT_100/"

# 2) MCLT case
TEST_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_test_mclt"
TRAIN_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_train_mclt"
LATENT_SPACE_SAVE_PATH = "/home/jaimelopez/TFM/datasets/fsdd/latent_space"
SAVE_DIR_GENERATED = "/home/jaimelopez/TFM/datasets/fsdd/"

# 3) Exponential case
# TEST_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_test"
# TRAIN_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_train"
# LATENT_SPACE_SAVE_PATH = "/home/jaimelopez/TFM/datasets/fsdd/latent_space"
# SAVE_DIR_GENERATED = "/home/jaimelopez/TFM/datasets/fsdd/latent_space_samples_exp/"

def parse_materials_name(val):
    if val == "01":
        return "Mesa de madera"
    elif val == "02":
        return "Base metálica"
    elif val == "03":
        return "Panel de madera"
    elif val == "040":
        return "Suelo: goma"
    elif val == "041":
        return "Suelo: directo"
    elif val == "042":
        return "Suelo: alfombrilla"
    else:
        return val

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 15))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        img = librosa.display.specshow(image, y_axis='log', sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time')
        ax.set(title='Original: Log-frequency power spectrogram')
        ax.label_outer()
        fig.colorbar(img, ax=ax, format="%+2.f dB")

        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        img = librosa.display.specshow(reconstructed_image, y_axis='log', sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time')
        ax.set(title='Reconstructed: Log-frequency power spectrogram')
        ax.label_outer()
        fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

def compute_mse(images, reconstructed_images):
    errors = []
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        reconstructed_image = reconstructed_image.squeeze()
        error = mean_squared_error(image, reconstructed_image, squared=False, multioutput='uniform_average')
        errors.append(error)
    array_errors = np.array(errors)
    MSE = np.mean(array_errors)
    return MSE

def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()

def get_files(file_paths):
    file_names = []
    for path in file_paths:
        file_name = os.path.basename(path)
        file_names.append(file_name)
    return file_names

def parse_labels(file_names, division="herramienta"):
    if division == "herramienta":
        labels = [i.split('#')[0] for i in file_names]
    elif division == "material":
        labels = [i.split('#')[1].split('.')[0] + "0" if ('04' in i and len(i.split('#')[1]) == 16) else
                 i.split('#')[1].split('.')[0] + i.split('#')[1][7] if '04' in i else i.split('#')[1].split('.')[0] for
                 i in file_names]
    else:
        labels = [i.split('-wa')[0] for i in file_names]
    label = preprocessing.LabelEncoder()
    labels = label.fit_transform(labels)
    mappings = dict(zip(range(len(label.classes_)), label.classes_))
    return labels, mappings

def plotly_latent_space(latent_representations, sample_labels, mappings, type="", save=False):
    layout = go.Layout(
        autosize=False,
        width=2500,
        height=2000
    )
    fig = go.Figure(layout=layout)
    length_latent_representations = latent_representations.shape[0]
    colors = []
    for i in range(80):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    for i in range(length_latent_representations):
        # print(latent_representations[:, i])
        fig.add_trace(go.Scatter(x=list(range(length_latent_representations)), y=latent_representations[i, :],
                                 mode='lines+markers',
                                 name=[mappings[j] for j in sample_labels][i],
                                 line=dict(color=colors[sample_labels[i]], width=2),
                                 legendgroup=[mappings[j] for j in sample_labels][i]
                                                                  ))
    if save:
        fig.write_image(LATENT_SPACE_SAVE_PATH + "/" + type + ".png")
    else:
        fig.show()

def plot_images_encoded_in_latent_space(latent_representations, sample_labels, mappings, type=""):
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=1,
                s=20)
    cbar = plt.colorbar(ticks=list(mappings.keys()))
    aux = 0
    cbar.ax.set_yticklabels([parse_materials_name(i) if type == "material" else i for i in mappings.values()])
   # for x, y in zip(latent_representations[:, 0],  latent_representations[:, 1]):
   #     label = f"({x},{y})"
   #     plt.annotate(list(mappings.keys())[sample_labels[aux]], (x,y), textcoords="offset points", xytext=(0,10), ha="center")
   #     aux += 1
    fig.show()
    # 1) STFT case
    # fig.savefig(LATENT_SPACE_SAVE_PATH + "/latent_space" + type + ".png")
    # 2) MCLT case
    # fig.savefig(LATENT_SPACE_SAVE_PATH + "/latent_space_mclt" + type + ".png")

def sample_latent_space(xmin=-15.0, xmax=5.0, ymin=-10.0, ymax=5.0, step=0.1):
    # x_axis = len(np.arange(xmin, xmax, step))
    # y_axis = len(np.arange(ymin, ymax, step))
    # axis = x_axis * y_axis

    latent_samples = []
    latent_file_names = []

    for x in np.arange(xmin, xmax, step):
        for y in np.arange(ymin, ymax, step):
            x_value = round(x,1)
            y_value = round(y,1)
            latent_samples.append([x_value,y_value])
            latent_file_names += [str(x_value)+'_'+str(y_value)]

    latent_samples = np.array(latent_samples)
    return latent_samples, latent_file_names

def generate(self, spectrograms):
    generated_spectrograms, latent_representations = \
        self.vae.reconstruct(spectrograms)
    signals = convert_spectrograms_to_audio(self, generated_spectrograms)
    return signals, latent_representations

def convert_spectrograms_to_audio(self, spectrograms, min=-60, max=10):
    signals = []
    limit = 5

    # for index, spectrogram in enumerate(spectrograms):
    for spectrogram in spectrograms:
        # reshape the log spectrogram
        log_spectrogram = spectrogram[:, :, 0]
        # apply denormalisation
        denorm_log_spec = self._min_max_normaliser.denormalise(
            log_spectrogram, min, max)
        # log spectrogram -> spectrogram
        spec = librosa.db_to_amplitude(denorm_log_spec)
        # apply Griffin-Lim
        signal = librosa.griffinlim(spec, hop_length=self.hop_length)
        # append signal to "signals"
        signals.append(signal)
        # if index == limit:
        #     break
    return signals

def save_latent_signals(signals, latent_file_names, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, latent_file_names[i] + ".wav")
        sf.write(save_path, signal, sample_rate)

def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


if __name__ == "__main__":
    vae = VAE.load("model")
    # vae = VAE_v2.load("model")
    x_test, file_paths = load_fsdd(TEST_PATH)
    x_train, file_paths = load_fsdd(TRAIN_PATH)

    # Get the file names from the test dataset and extract labels and mappings.
    file_names = get_files(file_paths)
    labels, mappings = parse_labels(file_names, division="herramienta")

    # Comparison of original spectrograms against reconstructed ones
    num_sample_images_to_show = 1
    sample_images, _ = select_images(x_test, labels, num_sample_images_to_show)
    # sample_images, _ = select_images(x_train, labels, num_sample_images_to_show)
    reconstructed_images, _ = vae.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)

    # Obtention of the MSE
    x_test_reconstructed, _ = vae.reconstruct(x_test)
    MSE = compute_mse(x_test, x_test_reconstructed)
    print("MSE: ",MSE)

    # Obtain the latent space representations and plot the latent space
    # _, latent_representations = vae.reconstruct(x_test)
    _, latent_representations = vae.reconstruct(x_train)
    # plot_images_encoded_in_latent_space(latent_representations, labels, mappings, type="")

    # Find the better fitted distribution to the dataset
    data_test = latent_representations[:,1]
    get_best_distribution(data=data_test)

    # # Get all the possible values of samples in a 2D latent space and reconstruct them on latent space
    # latent_samples, latent_file_names = sample_latent_space(step=1.0)
    #
    # # Scatter plot to show the distribution of the latent space samples across the axes
    # # fig = plt.figure(figsize=(10, 10))
    # # plt.scatter(latent_samples[:, 0],
    # #             latent_samples[:, 1],
    # #             cmap="rainbow",
    # #             alpha=1,
    # #             s=0.5)
    # # fig.show()
    #
    # reconstructed_images = VAE_v2.reconstruct_on_latent(vae, latent_samples)
    #
    # # Initialize Sound generator
    # sound_generator = SoundGenerator(vae, HOP_LENGTH)
    #
    # # generate audio for sampled spectrograms
    # signals, _ = generate(sound_generator, reconstructed_images)
    #
    # # save audio signals
    # save_latent_signals(signals, latent_file_names, SAVE_DIR_GENERATED)
