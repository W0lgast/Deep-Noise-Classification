from material.dataset import UrbanSound8KDataset
from random import randrange
from matplotlib import pyplot as plt

LABEL_NAMES = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}

LOGMELSPEC = 'logmelspec'
MFCC = 'mfcc'
CHROMA = 'chroma'
SPEC_CONT = 'spectral_contrast'
TONNETZ = 'tonnetz'


def display_lmc_data(data_features):
    lmc_data = {'LM': data_features[LOGMELSPEC],
                'Chroma': data_features[CHROMA],
                'Spectral Contrast': data_features[SPEC_CONT],
                'Tonnetz': data_features[TONNETZ]}

    widths = [max(s.shape[1] for s in lmc_data.values())]
    heights = [s.shape[0] for s in lmc_data.values()]

    fig, ax = plt.subplots(4, 1, constrained_layout=True,
                           gridspec_kw={'width_ratios': widths,
                                        'height_ratios': heights,
                                        'wspace': 0.025,
                                        'hspace': 0.05})

    for data, ax_i, label in zip(lmc_data.values(), ax, lmc_data.keys()):
        ax_i.imshow(data)
        ax_i.tick_params(left=False, labelleft=False)
        ax_i.get_xaxis().set_visible(False)
        ax_i.set_ylabel(label, rotation='horizontal',
                        ha='right', va='center', size=10)

    plt.show()


def display_mc_data(data_features):
    lmc_data = {'MFCC': data_features[MFCC],
                'Chroma': data_features[CHROMA],
                'Spectral Contrast': data_features[SPEC_CONT],
                'Tonnetz': data_features[TONNETZ]}

    widths = [max(s.shape[1] for s in lmc_data.values())]
    heights = [s.shape[0] for s in lmc_data.values()]

    fig, ax = plt.subplots(4, 1, constrained_layout=True,
                           gridspec_kw={'width_ratios': widths,
                                        'height_ratios': heights,
                                        'wspace': 0.025,
                                        'hspace': 0.05})

    for data, ax_i, label in zip(lmc_data.values(), ax, lmc_data.keys()):
        ax_i.imshow(data)
        ax_i.tick_params(left=False, labelleft=False)
        ax_i.get_xaxis().set_visible(False)
        ax_i.set_ylabel(label, rotation='horizontal',
                        ha='right', va='center', size=10)

    plt.show()

def display_mlmc_data(data_features):
    lmc_data = {'LM': data_features[LOGMELSPEC],
                'MFCC': data_features[MFCC],
                'Tonnetz': data_features[TONNETZ]}

    widths = [max(s.shape[1] for s in lmc_data.values())]
    heights = [s.shape[0] for s in lmc_data.values()]

    fig, ax = plt.subplots(3, 1, constrained_layout=True,
                           gridspec_kw={'width_ratios': widths,
                                        'height_ratios': heights,
                                        'wspace': 0.025,
                                        'hspace': 0.05})

    for data, ax_i, label in zip(lmc_data.values(), ax, lmc_data.keys()):
        ax_i.imshow(data)
        ax_i.tick_params(left=False, labelleft=False)
        ax_i.get_xaxis().set_visible(False)
        ax_i.set_ylabel(label, rotation='horizontal',
                        ha='right', va='center', size=10)

    plt.show()

dataset = UrbanSound8KDataset("material/UrbanSound8K_test.pkl", "MLMC")

random_segment = randrange(len(dataset))
print(random_segment)
data = dataset[random_segment]

print(LABEL_NAMES[data[1]])
print(data[2])

data = dataset.get_feature_dict(random_segment)

display_lmc_data(data)
display_mc_data(data)
display_mlmc_data(data)
