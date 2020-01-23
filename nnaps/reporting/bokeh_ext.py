
import pandas as pd

from sklearn import preprocessing, metrics

from astropy.stats import histogram as astro_hist

from bokeh import plotting as bpl
from bokeh import models as mpl
from bokeh.layouts import gridplot
from bokeh.transform import transform
from bokeh.io import output_file, show


def histogram(fig, data, bins='knuth', normalize=False, **kwargs):

    quad_kwargs = dict(fill_color="navy", line_color="white", alpha=0.5)
    quad_kwargs.update(kwargs)

    hist, edges = astro_hist(data, bins=bins)
    if normalize:
        hist = hist / len(data)
    fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], **quad_kwargs)

    return hist, edges, fig


def confusion_matrix(y_true, y_pred):
    plot_width, plot_height = 600, 500

    # based on https://stackoverflow.com/questions/49135741/bokeh-heatmap-from-pandas-confusion-matrix

    labels = y_true.unique()

    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels, normalize='true') * 100.

    df = pd.DataFrame(data=cm, columns=labels, index=labels)

    print(df)

    df.index.name = 'True'
    df.columns.name = 'Prediction'

    # Prepare data.frame in the right format
    df = df.stack().rename("value").reset_index()
    df['label'] = df['value'].apply(lambda x: "{:0.0f}".format(x))
    source = mpl.ColumnDataSource(df)

    # You can use your own palette here
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']

    # Had a specific mapper to map color with value
    mapper = mpl.LinearColorMapper(palette='Viridis256', low=df.value.min(), high=df.value.max())
    # Define a figure
    p = bpl.figure(
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=list(labels),
        y_range=list(labels),
        # toolbar_location=None,
        tools="",
    )
    # Create rectangle for heatmap
    r = p.rect(
        x="Prediction",
        y="True",
        width=1,
        height=1,
        source=source,
        line_color=None,
        fill_color=transform('value', mapper))

    p.xaxis.axis_label = 'Predicted values'
    p.yaxis.axis_label = 'True values'

    text_props = {"source": source, "text_align": "center", "text_baseline": "middle", "text_color":'white'}
    p.text(x='Prediction', y="True", text="label", **text_props)

    p.add_tools(mpl.HoverTool(tooltips=[("value", "@value")], mode="mouse",
                              point_policy="follow_mouse", renderers=[r]))

    # Add legend
    color_bar = mpl.ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        ticker=mpl.BasicTicker(desired_num_ticks=len(colors)))

    p.add_layout(color_bar, 'right')

    return p


def scatter_grid(data, parameters):
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

                hist, edges, p = histogram(p, data[xpar], bins='knuth')
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