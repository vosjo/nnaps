
from bokeh import plotting as bpl
from bokeh.layouts import gridplot, layout
from bokeh.models.widgets import Div

from nnaps.reporting import bokeh_ext


def make_training_history_report(predictor, filename=None):
    plot_width, plot_height = 600, 300

    history = predictor.history.copy()
    history['epoch'] = history.index

    targets = predictor.regressors + predictor.classifiers

    # sort targets in mae and accuracy evaluated variables: regressors and classifiers
    mae_targets, accuracy_targets = [], []
    for target in targets:
        if target + '_mae' in history.columns:
            mae_targets.append(target)
        if target + '_accuracy' in history.columns:
            accuracy_targets.append(target)

    def plot_metric_and_loss(history, target, metric='mae'):
        ykey = target + '_' + metric

        p1 = bpl.figure(plot_height=plot_height, plot_width=plot_width, title=target)
        p1.line(history['epoch'], history[ykey], legend_label='training')
        if 'val_' + ykey in history.columns:
            p1.line(history['epoch'], history['val_' + ykey], color='orange', legend_label='validation')

        # p1.xaxis.axis_label = 'Epoch'
        p1.yaxis.axis_label = 'MAE' if metric == 'mae' else 'Accuracy'
        p1.legend.location = "top_right" if metric == 'mae' else "bottom_right"
        p1.title.align = 'center'
        p1.title.text_font_size = '14pt'

        ykey = target + '_loss'
        p2 = bpl.figure(plot_height=plot_height, plot_width=plot_width)
        p2.line(history['epoch'], history[ykey])
        if 'val_' + ykey in history.columns:
            p2.line(history['epoch'], history['val_' + ykey], color='orange')

        p2.xaxis.axis_label = 'Epoch'
        p2.yaxis.axis_label = 'Loss'

        return gridplot([[p1], [p2]], toolbar_location='right')

    # make the figure
    bpl.output_file(filename, title='Training history')

    title_div = Div(text="""<h1>Training History</h1>""", width=1000, height=50)

    section_plots = [[title_div]]

    section_div = Div(text="""<h2>Regressors</h2>""", width=400, height=40)
    section_plots.append([section_div])

    mae_plots = []
    for target in mae_targets:
        p = plot_metric_and_loss(history, target, metric='mae')
        mae_plots.append(p)

    section_plots.append(mae_plots)

    section_div = Div(text="""<h2>Classifiers</h2>""", width=400, height=40)
    section_plots.append([section_div])

    acc_plots = []
    for target in accuracy_targets:
        p = plot_metric_and_loss(history, target, metric='accuracy')
        acc_plots.append(p)

    section_plots.append(acc_plots)

    cm_plots = []
    y_true = predictor.train_data[predictor.classifiers]
    y_pred = predictor.predict(predictor.train_data)[predictor.classifiers]
    for target in accuracy_targets:
        p_ = bokeh_ext.confusion_matrix(y_true[target], y_pred[target])
        p_.title.text = target
        p_.title.align = 'center'
        p_.title.text_font_size = '14pt'
        cm_plots.append([p_])

    section_plots.append(cm_plots)

    p = layout(section_plots)

    bpl.save(p)


def make_scaled_feature_plot(data, parameters, scalers):
    plot_width, plot_height = 300, 300

    org_plots = []
    for xpar in parameters:
        # make a histogram
        p = bpl.figure(plot_width=plot_width, plot_height=plot_height, title='original')
        bokeh_ext.histogram(p, data[xpar])
        p.xaxis.axis_label = xpar
        p.title.align = 'center'

        org_plots.append(p)

    scl_plots = []
    for xpar in parameters:
        # make a histogram
        if xpar in scalers and scalers[xpar] is not None:

            p = bpl.figure(plot_width=plot_width, plot_height=plot_height,
                           title='scaled: {}'.format(scalers[xpar].__class__.__name__))

            scaler = scalers[xpar]
            scaled_data = scaler.transform(data[[xpar]])

            bokeh_ext.histogram(p, scaled_data, fill_color="green")

            p.xaxis.axis_label = xpar
            p.title.align = 'center'

        else:
            p = bpl.figure(plot_width=plot_width, plot_height=plot_height,
                           title='No Scaler')
            p.title.align = 'center'

        scl_plots.append(p)

    return gridplot([org_plots, scl_plots])

def make_training_test_set_plot(train_data, test_data, features, regressors, classifiers):
    plot_width, plot_height = 300, 300

    def make_double_histogram(train_data, test_data):
        p = bpl.figure(plot_width=plot_width, plot_height=plot_height, title=par)

        hist, edges = bokeh_ext.histogram(p, train_data, bins='knuth', normalize=True, fill_color="blue", legend_label='train')
        hist, edges = bokeh_ext.histogram(p, test_data, bins=edges,fill_color="green", legend_label='test')

        p.title.align = 'center'

        return p

    div = Div(text="""<h3>Features</h3>""", width=1200, height=40)
    grid = [[div]]

    plots = []
    for par in features:
        p = make_double_histogram(train_data[[par]], test_data[[par]])
        plots.append([p])
    grid.append(plots)


    div = Div(text="""<h3>Regressors</h3>""", width=1200, height=40)
    grid.append([[div]])

    plots = []
    for par in regressors:
        p = make_double_histogram(train_data[[par]], test_data[[par]])
        plots.append([p])
    grid.append(plots)

    return layout(grid)


def make_training_data_report(predictor, filename=None):

    train_data = predictor.train_data
    test_data = predictor.test_data
    features = predictor.features
    regressors = predictor.regressors
    classifiers = predictor.classifiers

    bpl.output_file(filename, title='Training data')

    grid = []

    div = Div(text="""<h1>Training data report</h1>""", width=1200, height=60)
    grid.append([div])

    # scatter grid of the features
    div = Div(text="""<h2>Feature distribution</h2>""", width=1200, height=40)
    grid.append([div])

    scatter_grid = bokeh_ext.scatter_grid(train_data, features)
    grid.append([scatter_grid])

    # scaling plot of features
    div = Div(text="""<h2>Feature scaling</h2>""", width=1200, height=40)
    grid.append([div])

    scaled_plot = make_scaled_feature_plot(train_data, features, predictor.processors)
    grid.append([scaled_plot])

    # distribution of the regressors
    div = Div(text="""<h2>Regressor distribution</h2>""", width=1200, height=40)
    grid.append([div])

    scaled_plot = make_scaled_feature_plot(train_data, regressors, predictor.processors)
    grid.append([scaled_plot])

    # comparison training vs test set
    div = Div(text="""<h2>Training - Test set comparison</h2>""", width=1200, height=40)
    grid.append([div])

    plot = make_training_test_set_plot(train_data, test_data, features, regressors, classifiers)
    grid.append([plot])

    figure = layout(grid)
    bpl.save(figure)
