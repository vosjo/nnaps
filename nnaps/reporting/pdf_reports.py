import pylab as pl
import numpy as np
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