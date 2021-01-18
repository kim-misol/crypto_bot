import numpy as np
from sklearn.model_selection import train_test_split


def data_split(df):
    '''
     np.split will split at 60% of the length of the shuffled array,
     then 80% of length (which is an additional 20% of data),
     thus leaving a remaining 20% of the data. This is due to the definition of the function.
     You can test/play with: x = np.arange(10.0), followed by np.split(x, [ int(len(x)*0.6), int(len(x)*0.8)])
    '''
    # produces a 60%, 20%, 20% split for training, validation and test sets.
    train_df, val_df, test_df = np.split(df, [int(.6 * len(df)), int(.8 * len(df))]).copy()
    return train_df, val_df, test_df


def _data_split(examples, labels, train_frac=0.6, random_state=None):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        examples, labels, train_size=train_frac, random_state=random_state)

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_tmp, Y_tmp, train_size=0.5, random_state=random_state)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
