 
from bokeh import plotting as bpl
from bokeh import models as mpl
from bokeh.layouts import gridplot, layout
from bokeh.models.widgets import Div

def plot_training_history_html(history, targets=None, filename=None):
   
   #sort targets in mae and accuracy evaluated variables: regressors and classifiers
   mae_targets, accuracy_targets = [], []
   for target in targets:
      if target+'_mae' in history.columns:
         mae_targets.append(target)
      if target+'_accuracy' in history.columns:
         accuracy_targets.append(target)
   
   # make the figure
   bpl.output_file(filename, title='Training history')
   
   title_div = Div(text="""<h1>Training History</h1>""", width=1000, height=50)
   
   section_plots = [[title_div]]
   
   for target in mae_targets + accuracy_targets:
      
      section_div = Div(text="""<h2>{}</h2>""".format(target), width=400, height=40)
      section_plots.append([section_div])
      
      ykey = target+'_mae' if target in mae_targets else target+'_accuracy'
      
      p = bpl.figure(plot_height=500, plot_width=600, title='metric')
      p.line(history['epoch'], history[ykey], legend_label='training')
      
      if 'val_'+ykey in history.columns:
         p.line(history['epoch'], history['val_'+ykey], color='orange', legend_label='validation')
         
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


def plot_training_data_html(data, filename=None):
   
   # make the figure
   bpl.output_file(filename, title='Training data')
   
   
   bpl.save(p)
