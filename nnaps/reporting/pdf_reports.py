import pylab as pl
import numpy as np
import itertools
from sklearn import metrics

def plot_training_history(predictor):
    history = predictor.history
    history['epoch'] = history.index

    regressors = predictor.regressors
    classifiers = predictor.classifiers

    ncol = 2 if len(regressors+classifiers) > 1 else 1
    nrow = int(np.ceil(len(regressors)/ncol) + np.ceil(len(classifiers)/ncol))

    i = 0 # assign in case there are no regressors.
    for i, regressor in enumerate(regressors):

        pl.subplot(nrow, ncol, i+1)
        pl.plot(history['epoch'], history['val_'+regressor+'_mae'], label='test', color='C1')
        pl.plot(history['epoch'], history[regressor+'_mae'], label='train', color='C0')
        pl.title(regressor)

        # only first plot gets a legend
        if i == 0:
            pl.legend(loc='best')

        # only place ylabels on each first plot in a row
        if i % ncol == 0:
            pl.ylabel('MAE')

        if np.ceil(i / ncol) == nrow:
            pl.xlabel('Epoch')

    # if regressors don't fill the entire line, skip places to start classifiers on new line.
    if len(regressors) > 0:
        i += len(regressors) % ncol + 1

    for j, classifier in enumerate(classifiers):

        pl.subplot(nrow, ncol, i+j+1)
        pl.plot(history['epoch'], history['val_'+classifier+'_accuracy'], label='test', color='C1')
        pl.plot(history['epoch'], history[classifier+'_accuracy'], label='train', color='C0')
        pl.title(classifier)

        # only first plot gets a legend, need to check again in case there are no regressors
        if i == 0:
            pl.legend(loc='best')

        # only place ylabels on each first plot in a row
        if (i+j) % ncol == 0:
            pl.ylabel('Accuracy')

        if np.ceil((i+j+1) / ncol) == nrow:
            pl.xlabel('Epoch')


def plot_confusion_matrix(predictor):

    classifiers = predictor.classifiers

    y_pred = predictor.predict(predictor.train_data)[classifiers]
    y_true = predictor.train_data[classifiers]

    ncol = 2 if len(classifiers) > 1 else 1
    nrow = int(np.ceil(len(classifiers)/ncol))

    for i, classifier in enumerate(classifiers):
        ax = pl.subplot(nrow, ncol, i+1)
        labels = y_true[classifier].unique()
        cm = metrics.confusion_matrix(y_true[classifier], y_pred[classifier], normalize='true', labels=labels)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax)
        pl.title(classifier)
        pl.gca().grid(False)


def plot_feature_range_comparison(predictor, data):

    features = predictor.features
    train_data = predictor.train_data

    xy_pairs = list(itertools.combinations(features, 2))

    ncol = 3 if len(xy_pairs) > 2 else len(xy_pairs)
    nrow = int(np.ceil(len(xy_pairs)/ncol))

    for i, pair in enumerate(xy_pairs):
        ax = pl.subplot(nrow, ncol, i+1)

        ax.hexbin(train_data[pair[0]], train_data[pair[1]], gridsize=15, cmap='viridis')
        ax.plot(data[pair[0]], data[pair[1]], '.r', alpha=0.5, ms=1)
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])


