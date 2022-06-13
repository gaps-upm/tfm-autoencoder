import numpy as np
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import utils

from autoencoder import VAE
from sklearn import preprocessing
from generate import load_fsdd
#from train import load_mnist

TRAIN_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_train"
TEST_PATH = "/home/jaimelopez/TFM/datasets/fsdd/spectrograms_test"
LATENT_SPACE_SAVE_PATH = "/home/jaimelopez/TFM/datasets/fsdd/latent_space"

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


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

def parse_labels(file_names, division="herramienta"):
    if division == "herramienta":
        labels = [i.split('#')[0] for i in file_names]
    elif division == "material":
        labels = [i.split('#')[1].split('-')[0] + "0" if ('04' in i and len(i.split('#')[1]) == 17) else
                  i.split('#')[1].split('-')[0] + i.split('#')[1][8] if '04' in i else i.split('#')[1].split('-')[0] for
                  i in file_names]
    else:
        labels = [i.split('-chu')[0] for i in file_names]
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
                                 # marker_color=sample_labels,
                                 # marker=dict(
                                 #     size=16,
                                 #     color=sample_labels,  # set color equal to a variable
                                 #     colorscale='Viridis',  # one of plotly colorscales
                                 #     showscale=True,
                                 # )
                                 ))
    #
    # data=go.Scatter(
    #     #x=latent_representations[:, 0],
    #     x=list(range(80)),
    #     y=[latent_representations[:, i] for i in range(80)],
    #     mode='markers',
    #     hovertemplate='%{text}',
    #     text=[mappings[i] for i in sample_labels],
    #     marker=dict(
    #         size=16,
    #         color=sample_labels,  # set color equal to a variable
    #         colorscale='Viridis',  # one of plotly colorscales
    #         showscale=True,
    #     )
    # ))
    if save:
        fig.write_image(LATENT_SPACE_SAVE_PATH + "/" + type + ".png")
    else:
        fig.show()


if __name__ == "__main__":
    autoencoder = VAE.load("model")
    x_train, _ = load_fsdd(TRAIN_PATH)
    x_test, file_paths = load_fsdd(TEST_PATH)

    y_test, mappings =parse_labels(file_paths)

    num_sample_images_to_show = 8
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)

    num_images = 80
    sample_images, sample_labels = select_images(x_test, y_test, num_images)
    _, latent_representations = autoencoder.reconstruct(sample_images)
    plotly_latent_space(latent_representations, sample_labels, mappings, type="ls", save=True)
