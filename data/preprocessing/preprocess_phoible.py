import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def preprocess(data='data/datasets/raw_data/phoible.csv'):
    """
    Preprocess the Phoible dataset and return a preprocessed pandas dataframe of features for each phoneme.
    """

    # Read in dataset
    df = pd.read_csv(data, dtype={'SpecificDialect': str, 'Allophones': str, 'Marginal': str, 'tone': str})

    # Select the English dataset, as we are pronouncing with this and drop unused columns
    df = df[df.InventoryID == 2252]
    df.drop(['InventoryID', 'Glottocode', 'ISO6393', 'LanguageName', 'SpecificDialect', 'GlyphID', 'Allophones', 'Marginal', 'SegmentClass', 'Source'], axis=1, inplace=True)

    # Match Phoible characters with ours, so they match up.
    df = character_matching(df)

    df.set_index('Phoneme', inplace=True)

    # Replace all +'s with 1's and all -'s with -1's. Set undecided values to 0.
    df.replace({'+': 1, '-': -1, '-,+': 0, '+,-': 0}, inplace=True)
    df = df.astype('int32')

    return df


def character_matching(df):
    """
    Match Phoible ipa characters with our ipa characters
    """
    df['Phoneme'] = df['Phoneme'].replace({'kʰ': 'k', 'pʰ': 'p', 'tʰ': 't', 'ɡ': 'g', 'd̠ʒ': 'ʤ', 't̠ʃ': 'tʃ'})
    return df


def visualise_phoible_differences(df):
    """
    Visualise the differences between different Phoible characters
    """

    # Get Phoible characters and number of characters
    vals = df.index.tolist()
    size = len(vals)

    # Create a matrix of all features for each Phoible character
    num_features = len(df.iloc[0])
    scores = np.zeros((size, num_features))
    for i in range(size):
        for j in range(num_features):
            scores[i, j] = df.loc[vals[i]].iloc[j]

    # Calculate the differences between each pair of Phoible characters
    diffs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            diffs[i, j] = diff(vals[i], vals[j])

    fig, ax = plt.subplots()    
    ax.matshow(scores)
    fig.suptitle("Phoible features for each character")
    ax.set_xlabel('Feature')
    ax.set_ylabel('Character')
    
    fig, ax = plt.subplots()    
    ax.matshow(diffs)
    fig.suptitle("Absolute difference for each character across all Phoible features")
    ax.set_xlabel('Character')
    ax.set_ylabel('Character')

    plt.show()


def diff(x, y):
    """Absolute difference between two df entries"""
    return abs(df.loc[x] - df.loc[y]).sum()


if __name__ == '__main__':
    df = preprocess()
    print(df.head())
    visualise_phoible_differences(df)
