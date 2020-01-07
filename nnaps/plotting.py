from bokeh import plotting as bpl
from bokeh import models as mpl
from bokeh.layouts import gridplot, layout
from bokeh.models.widgets import Div

from astropy.stats import histogram

from sklearn import preprocessing


def plot_training_history_html(history, targets=None, filename=None):
    # sort targets in mae and accuracy evaluated variables: regressors and classifiers
    mae_targets, accuracy_targets = [], []
    for target in targets:
        if target + '_mae' in history.columns:
            mae_targets.append(target)
        if target + '_accuracy' in history.columns:
            accuracy_targets.append(target)

    # make the figure
    bpl.output_file(filename, title='Training history')

    title_div = Div(text="""<h1>Training History</h1>""", width=1000, height=50)

    section_plots = [[title_div]]

    for target in mae_targets + accuracy_targets:

        section_div = Div(text="""<h2>{}</h2>""".format(target), width=400, height=40)
        section_plots.append([section_div])

        ykey = target + '_mae' if target in mae_targets else target + '_accuracy'

        p = bpl.figure(plot_height=500, plot_width=600, title='metric')
        p.line(history['epoch'], history[ykey], legend_label='training')

        if 'val_' + ykey in history.columns:
            p.line(history['epoch'], history['val_' + ykey], color='orange', legend_label='validation')

        p.xaxis.axis_label = 'Epoch'

        if target in mae_targets:
            p.yaxis.axis_label = 'MAE'
            p.legend.location = "top_right"
        else:
            p.yaxis.axis_label = 'Accuracy'
            p.legend.location = "bottom_right"

        section_plots.append([p])

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


def plot_training_data_html(data, Xpars, filename=None):
    # make the figure
    bpl.output_file(filename, title='Training data')

    gp = make_scatter_grid_plot(data, Xpars)

    bpl.save(gp)
