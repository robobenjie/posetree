import plotly.graph_objs as go
import numpy as np
from contextlib import contextmanager

@contextmanager
def plotly_figure(title):
    layout = go.Layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')))

    fig = go.Figure(layout=layout)
    yield fig


    x_data = []
    y_data = []
    z_data = []

    for trace in fig.data:
        x_data.extend(trace['x'])
        y_data.extend(trace['y'])
        z_data.extend(trace['z'])

    x_min, x_max = [min(x_data), max(x_data)]
    y_min, y_max = [min(y_data), max(y_data)]
    z_min, z_max = [min(z_data), max(z_data)]

    # Set padding
    padding = 1.0  # change this value as needed

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[x_min - padding, x_max + padding], autorange=False),
            yaxis=dict(range=[y_min - padding, y_max + padding], autorange=False),
            zaxis=dict(range=[z_min - padding, z_max + padding], autorange=False),
            aspectmode='data'
        )
    )

    fig.show()

def plot_pose(fig, pose, name, length=0.35):
    x0, y0, z0 = pose.transform.position

    # The end points of the axis lines in the world coordinate frame
    x_end = pose.position + pose.x_axis * length
    y_end = pose.position + pose.y_axis * length
    z_end = pose.position + pose.z_axis * length


    # Create the axis lines
    x_line = go.Scatter3d(x=[x0, x_end[0]], y=[y0, x_end[1]], z=[z0, x_end[2]], mode='lines', line=dict(color='red'))
    y_line = go.Scatter3d(x=[x0, y_end[0]], y=[y0, y_end[1]], z=[z0, y_end[2]], mode='lines', line=dict(color='green'))
    z_line = go.Scatter3d(x=[x0, z_end[0]], y=[y0, z_end[1]], z=[z0, z_end[2]], mode='lines', line=dict(color='blue'))
    midpoint = pose.translate(.0 * np.array([length, length, length]))
    text_trace = go.Scatter3d(
        x=[midpoint.x],
        y=[midpoint.y],
        z=[midpoint.z],
        mode='text',
        text=[name],
        textfont=dict(
            color='black',
            size=10,
        )
    )

    for trance in [x_line, y_line, z_line, text_trace]:
        fig.add_trace(trance)