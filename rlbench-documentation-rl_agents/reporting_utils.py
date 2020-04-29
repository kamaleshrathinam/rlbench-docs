from metaflow import Flow,get_metadata,Run
print("Metadata",get_metadata())
from metaflow_train import FinalData
from typing import List
import chart_studio.plotly as py
import plotly.graph_objects  as go
import plotly.express as ps
from plotly.subplots import make_subplots
import math
import os 
import datetime
import itertools
import seaborn as sns
from scipy.stats import norm

def get_key_map(arr):
    finalmap = []
    for i in itertools.product(*arr):
        finalmap.append(i)
    return finalmap

def plot_and_save_grad_figures(run:Run):
    """
    Directly plots gradients. Will Die if gradients are too large. 
    """
    final_data_arr = run.data.final_data
    run_dir_path = "RunAnalytics/Run-"+run.id
    # Gradients are collected for whole Flow. So if 1st doesnt have. No one has. 
    if not hasattr(final_data_arr[0],'gradients'): 
        return None
    if len(final_data_arr[0].gradients['avg']) == 0:
        return None 
    last_time = datetime.datetime.now()
    for data in final_data_arr:
        fig = go.Figure()
        gradients = data.gradients
        for i in range(len(gradients['avg'])):
            avg_grads = gradients['avg'][i]
            layers = gradients['layer'][i]
            fig.add_trace(go.Scatter(
                        x=layers, \
                        y=avg_grads, \
                        line=dict(color='blue',width=1),
                        opacity=0.8))

            curr_time = datetime.datetime.now()
            if (curr_time - last_time).total_seconds() > 30:
                print("Completed %d Percentage of injesting to Chart "% ((i/len(data.gradients['avg']))*100))
                last_time = curr_time
        
        fig.update_xaxes(tickmode ='array',tickvals = layers,ticktext = layers,tickangle=90,range=[0, len(avg_grads)])
        fig.update_layout(title_text="Gradient Flows of "+data.agent_name,height=1000,showlegend=False,width=1000)
        fig.write_image(run_dir_path+"/gradient_"+data.agent_name+".png")

def make_consolidated_loss_plot(final_data_arr:List[FinalData])->go.Figure:
    """
    Plots loss of all agents in one Figure and returns the figure. 
    """
    loss_plot = go.Figure(layout=dict(title=dict(text="Plot of Running Losses of All Models")))
    legend = []
    for i in range(len(final_data_arr)):
        loss_op = final_data_arr[i].loss
        if loss_op is None:
            continue
        agent_name = final_data_arr[i].agent_name
        epochs = [j+1 for j in range(len(loss_op))]
        loss_plot.add_trace(go.Scatter(
                    x=epochs,
                    y=loss_op,
                    name=agent_name,
                    opacity=0.8))

    loss_plot.update_layout(xaxis_title="Epochs",
        yaxis_title="Loss")
    return loss_plot

def plot_grad_figures(final_data_arr:List[FinalData])->go.Figure:
    """
    Plots Figures for Gradients in a Single Figure. 
    
    WARNING : SAVING AND PRINTING GRADIENTS IS HEAVY
    """
    if not hasattr(final_data_arr[0],'gradients'): # Gradients are collected for whole Flow. So if 1st doesnt have. No one has. 
            return None
    if len(final_data_arr[0].gradients['avg']) == 0:
        return None 
    rows = math.ceil(len(final_data_arr)/2)
    last_time = datetime.datetime.now()
    index_queue = get_key_map([[i+1 for i in range(rows)],[1,2]]) # Make rows and columns. 
    fig = make_subplots(rows=rows, cols=2, start_cell="bottom-left",subplot_titles=[data.agent_name for data in final_data_arr])
    for data in final_data_arr:
        gradients = data.gradients
        row,col = index_queue.pop(0)
        
        for i in range(len(gradients['avg'])):
            avg_grads = gradients['avg'][i]
            layers = gradients['layer'][i]
            fig.add_trace(go.Scatter(
                        x=layers, \
                        y=avg_grads, \
                        line=dict(color='blue',width=1),
                        opacity=0.8),row=row,col=col)

            curr_time = datetime.datetime.now()
            if (curr_time - last_time).total_seconds() > 30:
                print("Completed %d Percentage of injesting to Chart "% ((i/len(data.gradients['avg']))*100))
                last_time = curr_time
        
        fig.update_xaxes(tickmode ='array',tickvals = layers,ticktext = layers,tickangle=90,range=[0, len(avg_grads)],row=row,col=col)
        fig.update_layout(title_text="Gradient Flows of Different Agents",height=2000,showlegend=False,width=2000)

    return fig

def make_loss_plots(final_data_arr:List[FinalData])->go.Figure:
    """
    Makes the Loss Plot of all Agents within one single Plot. 
    """
    rows = math.ceil(len(final_data_arr)/2)+1
    last_time = datetime.datetime.now()
    index_queue = get_key_map([[i+1 for i in range(rows)],[1,2]]) # Make rows and columns. 
    last_row,last_col = index_queue[-1]
    loss_plot = make_subplots(
        rows=rows,\
        cols=2,\
        subplot_titles=[data.agent_name for data in final_data_arr]+["All Losses In One Plot"], \
        specs=[ [{}, {}] for _ in range(rows-1) ]+ [[ {"colspan": 2}, None] ]
    )
    for data in final_data_arr:
        row,col = index_queue.pop(0)
        loss_op = data.loss
        if not isinstance(loss_op,list):
            continue
        agent_name = data.agent_name
        epochs = [j+1 for j in range(len(loss_op))]
        loss_plot.add_trace(go.Scatter(
                    x=epochs,
                    y=loss_op,
                    name=agent_name,
                    line=dict(width=1),
                    opacity=0.8),row=row,col=col)

        loss_plot.add_trace(go.Scatter(
                    x=epochs,
                    y=loss_op,
                    name=agent_name,
                    line=dict(width=1),
                    opacity=0.8),row=last_row,col=1)

        loss_plot.update_yaxes(title_text="Loss",row=row,col=col)
        loss_plot.update_xaxes(title_text="Epochs",row=row,col=col)
    
    loss_plot.update_yaxes(title_text="Loss",row=last_row,col=1)
    loss_plot.update_xaxes(title_text="Epochs",row=last_row,col=1)
    loss_plot.update_layout(title_text="Plot of Running Losses of All Models",height=2000,showlegend=True,width=2000)
    return loss_plot


def make_convergence_plots(final_data_arr:List[FinalData])->go.Figure:
    """
    Plots the Convergence metrics collected of each model
    FinalData Holds this data. 
    """
    rows = math.ceil(len(final_data_arr)/2)+1
    last_time = datetime.datetime.now()
    index_queue = get_key_map([[i+1 for i in range(rows)],[1,2]]) # Make rows and columns. 
    last_row,_ = index_queue[-1]
    convergence_plot = make_subplots(
        rows=rows,\
        cols=2,\
        subplot_titles=[data.agent_name for data in final_data_arr]+["Convergence Comparisons"], \
        specs=[ [{}, {}] for _ in range(rows-1) ]+ [[ {"colspan": 2}, None] ]
    )
    agent_names = []
    num_convergences = []
    for data in final_data_arr:
        row,col = index_queue.pop(0)
        simulation_analytics = data.simulation_analytics
        if not isinstance(simulation_analytics,dict):
            continue
        agent_names.append(data.agent_name)
        steps_to_convergence_of_agent = list(map(lambda x: x['steps_to_convergence'],simulation_analytics['convergence_metrics']))
        step_to_convergence,frequency_of_occurence = CountFrequency(steps_to_convergence_of_agent)
        num_convergences.append(len(simulation_analytics['convergence_metrics']))
        convergence_plot.add_trace(go.Scatter(
                    x=step_to_convergence,
                    y=frequency_of_occurence,
                    mode="markers",
                    opacity=0.8),row=row,col=col)

        convergence_plot.update_yaxes(title_text="Frequency",row=row,col=col)
        convergence_plot.update_xaxes(title_text="Num Steps",row=row,col=col)
    
    convergence_plot.add_trace(go.Bar(x=agent_names,y=num_convergences),row=last_row,col=1)
    convergence_plot.update_yaxes(title_text="Number Of Convergences",row=last_row,col=1)
    convergence_plot.update_xaxes(title_text="Models Name",row=last_row,col=1)
    convergence_plot.update_layout(title_text="Convergence Distributions For Different Agents",height=2000,showlegend=False,width=2000)
    return convergence_plot

def CountFrequency(my_list):       
    # Creating an empty dictionary  
    freq = {} 
    for items in my_list: 
        freq[items] = my_list.count(items) 
    x ,y=[],[]
    for key, value in freq.items(): 
        x.append(key)
        y.append(value)
    return x,y

def save_main_run_data(run:Run):
    """
    Expects the `final_data` property as a part of the Run.
    """
    run_dir_path = "RunAnalytics/Run-"+run.id
    print("Saving to Path :",run_dir_path)
    safe_mkdir(run_dir_path)
    write_file = open(run_dir_path+"/model_results.txt",'w')
    l = make_loss_plots(run.data.final_data)
    l.write_image(run_dir_path+"/model_losses.png")
    print("Saved Loss To to Path :",run_dir_path+"/model_losses.png")
    c = make_convergence_plots(run.data.final_data)
    c.write_image(run_dir_path+"/model_convergence.png")
    print("Saved Loss To to Path :",run_dir_path+"/model_convergence.png")
    write_file.writelines('\n'.join([str(data) for data in run.data.final_data]))
    write_file.close()

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def get_run_stats(min_num_epochs=100,min_demos=100):
    save_objs = [] 
    for run in Flow('TrainingSimulatorFlow').runs():
        if not run.finished:
            continue
        flow_init_datum = list(run.steps())[-1].task.data 
        if flow_init_datum.num_demos >= min_demos and flow_init_datum.num_epochs >= min_num_epochs: 
            nw_objs = [data.to_json()for data in run.data.final_data] # capture flows > 100 demos/ 100 epochs. 
            save_objs = save_objs + nw_objs
    
    return save_objs


def make_distance_plots(final_data_arr: List[FinalData]) -> go.Figure:

    for data in final_data_arr:
        simulation_analytics = data.simulation_analytics
        if not isinstance(simulation_analytics, dict):
            continue
        agent_name=(data.agent_name)
        distances=list(
            map(lambda x: x['distance'], simulation_analytics['distance_metrics']))

    distance_plot=sns.distplot(distances, bins=50, kde=False)
    #distance_plot=sns.distplot(distances, fit=norm, kde=False)
    distance_plot.set(xlabel='Distances', ylabel='Frequency')

    return distance_plot