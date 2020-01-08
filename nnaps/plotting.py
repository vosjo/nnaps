from bokeh import plotting as bpl
from bokeh import models as mpl
from bokeh.layouts import gridplot, layout
from bokeh.models.widgets import Div

from astropy.stats import histogram

from sklearn import preprocessing


def plot_training_history_html(history, targets=None, filename=None):
    plot_width, plot_height = 600, 300

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

    p = layout(section_plots)

    bpl.save(p)


def make_scatter_grid_plot(data, parameters):
    plot_width, plot_height = 300, 300

    data = data[parameters]

    # add scaled data for the hexbin plots
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    for i, p in enumerate(parameters):
        data[p + 'scaled'] = scaled_data[:, i]

    # new formatters so that hexbin plots show the original range of data on axis instead of [0-1]
    formatters = {}
    for p in parameters:
        vmin, vmax = data[p].min(), data[p].max()

        code = """
      tick = {vmin} + tick * ({vmax} - {vmin})
      return tick.toFixed(1)
      """.format(vmin=vmin, vmax=vmax)

        formatters[p] = mpl.FuncTickFormatter(code=code)

    source = mpl.ColumnDataSource(data)

    grid_plots = []
    for i, xpar in enumerate(parameters):
        line_plots = []
        for j, ypar in enumerate(parameters):

            if i == j:
                # make a histogram
                p = bpl.figure(plot_width=plot_width, plot_height=plot_height)

                hist, edges = histogram(data[xpar], bins='knuth')
                p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                       fill_color="navy", line_color="white", alpha=0.5)

                p.xaxis.axis_label = xpar
            elif i < j:
                # make a scatter plot
                p = bpl.figure(plot_width=plot_width, plot_height=plot_height)
                p.scatter(xpar, ypar, source=source, alpha=0.5)
                p.xaxis.axis_label = xpar
                p.yaxis.axis_label = ypar
            else:
                # make a hex plot
                p = bpl.figure(plot_width=plot_width, plot_height=plot_height)

                r, bins = p.hexbin(data[xpar + 'scaled'], data[ypar + 'scaled'], size=0.05)
                p.xaxis.axis_label = xpar
                p.yaxis.axis_label = ypar
                p.xaxis.formatter = formatters[xpar]
                p.yaxis.formatter = formatters[ypar]

                p.add_tools(mpl.HoverTool(tooltips=[("count", "@c")], mode="mouse",
                                          point_policy="follow_mouse", renderers=[r]))

            line_plots.append(p)

        grid_plots.append(line_plots)

    return gridplot(grid_plots)


def make_scaled_feature_plot(data, parameters, scalers):
    plot_width, plot_height = 300, 300

    org_plots = []
    for xpar in parameters:
        # make a histogram
        p = bpl.figure(plot_width=plot_width, plot_height=plot_height, title='original')

        hist, edges = histogram(data[xpar], bins='knuth')
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="navy", line_color="white", alpha=0.5)

        p.xaxis.axis_label = xpar
        p.title.align = 'center'

        org_plots.append(p)

    scl_plots = []
    for xpar in parameters:
        # make a histogram
        if xpar in scalers:
            p = bpl.figure(plot_width=plot_width, plot_height=plot_height,
                           title='scaled: {}'.format(scalers[xpar].__name__))

            scaler = scalers[xpar]()
            scaled_data = scaler.fit_transform(data[[xpar]])

            hist, edges = histogram(scaled_data, bins='knuth')
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                   fill_color="green", line_color="white", alpha=0.5)

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

        hist, edges = histogram(train_data, bins='knuth')
        hist = hist / len(train_data)
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="blue", line_color="white", alpha=0.5, legend_label='train')

        hist, edges = histogram(test_data, bins=edges)
        hist = hist / len(test_data)
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="green", line_color="white", alpha=0.5, legend_label='test')

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


def plot_training_data_html(train_data, test_data, Xpars, regressors, classifiers, processors, filename=None):
    # make the figure
    bpl.output_file(filename, title='Training data')

    grid = []

    div = Div(text="""<h1>Training data report</h1>""", width=1200, height=60)
    grid.append([div])

    # scatter grid of the features
    div = Div(text="""<h2>Feature distribution</h2>""", width=1200, height=40)
    grid.append([div])

    scatter_grid = make_scatter_grid_plot(train_data, Xpars)
    grid.append([scatter_grid])

    # scaling plot of features
    div = Div(text="""<h2>Feature scaling</h2>""", width=1200, height=40)
    grid.append([div])

    scaled_plot = make_scaled_feature_plot(train_data, Xpars, processors)
    grid.append([scaled_plot])

    # distribution of the regressors
    div = Div(text="""<h2>Regressor distribution</h2>""", width=1200, height=40)
    grid.append([div])

    scaled_plot = make_scaled_feature_plot(train_data, regressors, processors)
    grid.append([scaled_plot])

    # comparison training vs test set
    div = Div(text="""<h2>Training - Test set comparison</h2>""", width=1200, height=40)
    grid.append([div])

    plot = make_training_test_set_plot(train_data, test_data, Xpars, regressors, classifiers)
    grid.append([plot])

    figure = layout(grid)
    bpl.save(figure)
