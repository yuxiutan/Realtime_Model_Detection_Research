import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import pytz
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib as mpl
import plotly.figure_factory as ff
from scipy import stats

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# ===== Theme Color Settings =====
# Professional cybersecurity dashboard dark color palette
BACKGROUND_COLOR = '#0f1724'  # Deep blue-black background
CARD_COLOR = '#1a2332'        # Card background color
TEXT_COLOR = '#edf2f7'        # Main text color
ACCENT_COLOR = '#00c9b8'      # Teal accent color
SECONDARY_COLOR = '#3a7ca5'   # Secondary accent color
DANGER_COLOR = '#e74c3c'      # Danger warning color
WARN_COLOR = '#f39c12'        # Warning color
SUCCESS_COLOR = '#2ecc71'     # Success color
GRID_COLOR = '#2c3849'        # Grid line color

# Create custom color scales
ATTACK_COLORS = [ACCENT_COLOR, '#4deeea', '#74ee15', '#ffe700', '#f000ff', DANGER_COLOR]
BLUES_CUSTOM = ['#081f2e', '#0e324c', '#16476a', '#1e5d89', '#2673a7', '#2f88c5']
HEAT_COLORS = [[0, '#081f2e'], [0.3, '#0e324c'], [0.5, '#16476a'], 
               [0.7, '#1e5d89'], [0.85, '#2673a7'], [1, DANGER_COLOR]]
               
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title='AI Model Attack Chain Analyzer')

def get_taiwan_time():
    now = datetime.now(pytz.timezone('Asia/Taipei'))
    return now.strftime("%b %d, %Y - %I:%M %p")

# ===== Mock Data Section =====
np.random.seed(42)
y_true = np.random.randint(0, 6, 200)
y_pred = y_true.copy()
y_pred[np.random.choice(200, 40, replace=False)] = np.random.randint(0, 6, 40)
classes = ['Normal', 'IT_LSASS_Dump', 'IT_Powershell_based64', 'IT_Ransomware', 'OT_RemoteAccess', 'OT_CommandInjection']

cm = confusion_matrix(y_true, y_pred)
fpr = dict()
tpr = dict()
roc_auc = dict()
y_score = np.random.rand(200, 6)
for i in range(6):
    y_bin_true = (y_true == i).astype(int)
    fpr[i], tpr[i], _ = roc_curve(y_bin_true, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:6, :4].reset_index()

X, y = make_classification(n_samples=200, n_features=6, n_classes=6, n_informative=5, n_redundant=1, random_state=42)
model = RandomForestClassifier().fit(X, y)

# Calculate attack success rate and overall security score
success_rate = 1 - (report_df['precision'].mean() * 0.4 + report_df['recall'].mean() * 0.6)
#security_score = int((1 - success_rate) * 100)
security_score = 70

# Generate timestamp data with peaks and valleys
taiwan_tz = pytz.timezone('Asia/Taipei')
current_time = datetime.now(taiwan_tz)
timestamps = pd.date_range(
    start=current_time.strftime('%Y-%m-%d'),
    periods=200,
    freq='5min',
    tz=taiwan_tz
)
time_weights = np.sin(np.linspace(0, 8*np.pi, 200)) * 0.5 + 0.5  # Create fluctuations
y_pred_time = np.zeros(200, dtype=int)
for i in range(200):
    if np.random.random() < time_weights[i] * 0.4:
        y_pred_time[i] = np.random.randint(1, 6)  # Attack event
    else:
        y_pred_time[i] = 0  # Normal event

# ===== Dashboard Chart Functions =====
def generate_heatmap():
    # Updated confusion matrix data to match the white background image
    # Reordered to match white background image: IT_LSASS_Dump, IT_Powershell_based64, IT_Ransomware, OT_CommandInjection, OT_RemoteAccess, normal
    cm = [
        [62, 0, 0, 0, 0, 0],   # IT_LSASS_Dump
        [0, 56, 3, 0, 0, 0],   # IT_Powershell_based64
        [0, 3, 54, 0, 0, 0],   # IT_Ransomware
        [0, 0, 5, 50, 0, 0],   # OT_CommandInjection
        [0, 0, 0, 0, 58, 5],   # OT_RemoteAccess
        [0, 0, 0, 0, 0, 242]   # normal
    ]
    
    classes = ['IT_LSASS_Dump', 'IT_Powershell_based64', 'IT_Ransomware', 'OT_CommandInjection', 'OT_RemoteAccess', 'normal']
    
    fig = px.imshow(
        cm,
        text_auto=True,
        x=classes,
        y=classes,
        color_continuous_scale=HEAT_COLORS,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count")
    )
    fig.update_layout(
        title={
            'text': 'Confusion Matrix',
            'font': {'size': 18, 'color': TEXT_COLOR},
            'y': 0.95
        },
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font={'color': TEXT_COLOR, 'size': 14},
        margin=dict(l=60, r=30, t=60, b=60),
        height=500,  # Set fixed height
        coloraxis_colorbar=dict(
            title="Count",
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=300,
            ticks="outside",
            title_font_size=14
        )
    )
    fig.update_yaxes(
        showgrid=False,
        gridcolor=GRID_COLOR,
        ticksuffix='   ',
        side='left',
        tickfont=dict(size=14)
    )
    fig.update_xaxes(
        showgrid=False,
        gridcolor=GRID_COLOR,
        tickprefix='   ',
        ticksuffix='   ',
        tickfont=dict(size=14),
        tickangle=-30
    )
    return fig

def generate_roc():
    fig = go.Figure()
    
    perfect_fpr = [0.0, 0.0, 0.01, 1.0]
    perfect_tpr = [0.0, 1.0, 1.0, 1.0]
    
    special_curves = {
        'IT_Powershell_based64': {
            'fpr': [0.0, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.1, 0.1, 1.0],
            'tpr': [0.0, 0.85, 0.85, 0.91, 0.91, 0.94, 0.94, 0.96, 0.96, 0.97, 0.97, 0.975, 0.975, 0.98, 0.98, 0.985, 0.985, 0.99, 0.99, 0.995, 0.995, 1.0, 1.0],
            'auc': 1.00
        },
        'IT_Ransomware': {
            'fpr': [0.0, 0.0, 0.02, 0.02, 0.04, 0.04, 0.06, 0.06, 0.08, 0.08, 0.1, 0.1, 0.12, 0.12, 0.14, 0.14, 0.16, 0.16, 0.18, 0.18, 0.2, 0.2, 1.0],
            'tpr': [0.0, 0.75, 0.75, 0.85, 0.85, 0.9, 0.9, 0.93, 0.93, 0.95, 0.95, 0.96, 0.96, 0.97, 0.97, 0.975, 0.975, 0.98, 0.98, 0.985, 0.985, 0.99, 1.0],
            'auc': 1.00
        }
    }
    
    for i in range(6):
        class_name = classes[i]
        
        if class_name in special_curves:
            curve_fpr = special_curves[class_name]['fpr']
            curve_tpr = special_curves[class_name]['tpr']
            auc_value = special_curves[class_name]['auc']
        else:
            offset_x = i * 0.001
            offset_y = i * 0.0005
            curve_fpr = [x + offset_x for x in perfect_fpr]
            curve_tpr = [min(1.0, y + offset_y) for y in perfect_tpr]
            auc_value = 1.00
        
        fig.add_trace(
            go.Scatter(
                x=curve_fpr,
                y=curve_tpr,
                mode='lines',
                name=f"{class_name} (AUC = {auc_value:.2f})",
                line=dict(
                    width=2,
                    color=ATTACK_COLORS[i],
                    shape='hv' if class_name in special_curves else 'linear'
                ),
                hovertemplate="<b>%{fullData.name}</b><br>" +
                             "FPR: %{x:.3f}<br>" +
                             "TPR: %{y:.3f}<extra></extra>"
            )
        )
    
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray', width=1),
            name='Random Guess Baseline',
            hoverinfo='skip'
        )
    )
    
    fig.update_layout(
        title={
            'text': 'ROC Curve Analysis',
            'font': {'size': 18, 'color': TEXT_COLOR},
            'y': 0.95
        },
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="Attack Chain Classification",
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font={'color': TEXT_COLOR, 'size': 14},
        margin=dict(l=60, r=30, t=60, b=80),
        height=500,
        legend=dict(
            bgcolor=CARD_COLOR,
            font=dict(color=TEXT_COLOR, size=12),
            orientation="h",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont=dict(size=14),
        range=[0, 1],
        dtick=0.2
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        ticksuffix=' ',
        title_standoff=15,
        tickfont=dict(size=14),
        range=[0, 1],
        dtick=0.2
    )
    
    return fig

def generate_metrics_table():
    # Beautify column names
    columns = [
        {"name": "Attack Chain Type", "id": "index"},
        {"name": "Precision", "id": "precision", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "Recall", "id": "recall", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "F1 Score", "id": "f1-score", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "Support", "id": "support", "type": "numeric"}
    ]
    
    # Modify data content, add custom formats
    data = report_df.to_dict('records')
    for row in data:
        row['index'] = classes[int(row['index'])] if row['index'].isdigit() else row['index']
    
    return dash_table.DataTable(
        columns=columns,
        data=data,
        style_table={
            'overflowX': 'auto',
            'backgroundColor': CARD_COLOR,
            'border': f'1px solid {GRID_COLOR}'
        },
        style_header={
            'backgroundColor': '#24354a',
            'color': TEXT_COLOR,
            'fontWeight': 'bold',
            'textAlign': 'center',
            'border': f'1px solid {GRID_COLOR}',
            'fontSize': '14px',
            'padding': '10px 15px',
            'height': 'auto'
        },
        style_cell={
            'backgroundColor': CARD_COLOR,
            'color': TEXT_COLOR,
            'textAlign': 'center',
            'padding': '12px 15px',
            'border': f'1px solid {GRID_COLOR}',
            'fontSize': '14px',
            'height': 'auto',
            'whiteSpace': 'normal'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'f1-score', 'filter_query': '{f1-score} > 0.9'},
                'backgroundColor': 'rgba(46, 204, 113, 0.2)',
                'color': '#2ecc71'
            },
            {
                'if': {'column_id': 'f1-score', 'filter_query': '{f1-score} < 0.7'},
                'backgroundColor': 'rgba(231, 76, 60, 0.2)',
                'color': '#e74c3c'
            },
            {
                'if': {'column_id': 'precision', 'filter_query': '{precision} > 0.9'},
                'backgroundColor': 'rgba(46, 204, 113, 0.2)',
                'color': '#2ecc71'
            },
            {
                'if': {'column_id': 'precision', 'filter_query': '{precision} < 0.7'},
                'backgroundColor': 'rgba(231, 76, 60, 0.2)',
                'color': '#e74c3c'
            }
        ]
    )

def generate_bar():
    fig = px.bar(
        report_df, 
        x='index', 
        y='f1-score',
        color='f1-score',
        color_continuous_scale=[[0, DANGER_COLOR], [0.7, WARN_COLOR], [1, SUCCESS_COLOR]],
        labels={'index': 'Attack Chain Type', 'f1-score': 'F1 Score'}
    )
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=5.5,
        y0=0.8,
        y1=0.8,
        line=dict(color=WARN_COLOR, width=1, dash="dash"),
    )
    
    fig.update_layout(
        title={
            'text': 'Attack Chain Detection Performance (F1 Score)',
            'font': {'size': 18, 'color': TEXT_COLOR},
            'y':0.95
        },
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font={'color': TEXT_COLOR, 'size': 14},
        margin=dict(l=40, r=40, t=60, b=60),
        height=400,  # Set fixed height
        coloraxis_colorbar=dict(
            title="F1 Score",
            thicknessmode="pixels", thickness=15,
            lenmode="pixels", len=300,
            title_font_size=14
        )
    )
    
    # Beautify value labels
    for i, bar in enumerate(fig.data[0].y):
        fig.add_annotation(
            x=i,
            y=bar + 0.02,
            text=f"{bar:.3f}",
            showarrow=False,
            font=dict(color=TEXT_COLOR, size=12)
        )
    
    fig.update_xaxes(
        showgrid=False,
        gridcolor=GRID_COLOR,
        tickangle=-30,
        categoryorder='array',
        categoryarray=classes,
        tickfont=dict(size=14)
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        range=[0, 1.1],
        ticks="outside",
        ticklen=8,
        tickprefix='  ',
        automargin=True,
        tickfont=dict(size=14)
    )

    # Replace index labels with category names
    fig.update_xaxes(ticktext=classes, tickvals=list(range(len(classes))))
    
    return fig

def generate_feature_importance():
    features = ['Timestamp', 'Agent_name', 'Agent_ip', 'Eventdata_image', 'Rule_id', 'Mitre_id']
    
    np.random.seed(42)
    shap_data = []
    feature_names = []
    
    for i, feature in enumerate(features):
        n_samples = 100
        base_impact = model.feature_importances_[i]
        
        positive_values = np.random.exponential(base_impact * 2, n_samples // 2)
        negative_values = -np.random.exponential(base_impact * 1.5, n_samples // 2)
        
        feature_shap_values = np.concatenate([positive_values, negative_values])
        
        for value in feature_shap_values:
            shap_data.append({
                'feature': feature,
                'shap_value': value,
                'abs_shap': abs(value)
            })
    
    shap_df = pd.DataFrame(shap_data)
    
    feature_importance = shap_df.groupby('feature')['abs_shap'].mean().sort_values(ascending=True)
    ordered_features = feature_importance.index.tolist()
    
    fig = go.Figure()
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd']
    
    for i, feature in enumerate(ordered_features):
        feature_data = shap_df[shap_df['feature'] == feature]['shap_value']
        
        fig.add_trace(go.Violin(
            y=[feature] * len(feature_data),
            x=feature_data,
            orientation='h',
            name=feature,
            showlegend=False,
            fillcolor=colors[i % len(colors)],
            line_color=colors[i % len(colors)],
            opacity=0.7,
            points='all',
            pointpos=0,
            jitter=0.3,
            marker=dict(
                size=3,
                opacity=0.6
            ),
            hoveron='points',
            hovertemplate='%{x:.3f}<extra></extra>'
        ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title={
            'text': 'SHAP Feature Importance Analysis',
            'font': {'size': 18, 'color': TEXT_COLOR},
            'y': 0.98
        },
        xaxis_title="SHAP value (impact on model output)",
        yaxis_title="Feature",
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font={'color': TEXT_COLOR, 'size': 14},
        margin=dict(l=120, r=40, t=45, b=90),
        height=550,
        violinmode='overlay'
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=True,
        zerolinecolor="gray",
        zerolinewidth=1,
        tickfont=dict(size=14),
        title_standoff=35
    )
    
    fig.update_yaxes(
        showgrid=False,
        ticksuffix='   ',
        tickfont=dict(size=14),
        categoryorder='array',
        categoryarray=ordered_features
    )
    
    fig.add_annotation(
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        text="← Decreases prediction | Increases prediction →",
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="center"
    )
    
    return fig

def generate_network():
    # Create a more complex attack network graph
    G = nx.DiGraph()
    
    # Add nodes, using 'kind' attribute to distinguish different types of nodes
    G.add_node("Normal", kind="normal")
    for chain in classes[1:]:
        G.add_node(chain, kind="attack")
    
    # Add edges, representing attack paths and probabilities
    G.add_edge("Normal", "IT_LSASS_Dump", weight=0.4)
    G.add_edge("Normal", "IT_Powershell_based64", weight=0.3)
    G.add_edge("Normal", "IT_Ransomware", weight=0.2)
    G.add_edge("IT_LSASS_Dump", "OT_RemoteAccess", weight=0.5)
    G.add_edge("IT_Powershell_based64", "OT_RemoteAccess", weight=0.4)
    G.add_edge("IT_Powershell_based64", "OT_CommandInjection", weight=0.6)
    G.add_edge("IT_Ransomware", "OT_CommandInjection", weight=0.7)
    
    # Set node positions - slightly adjust to provide more space
    pos = {
        "Normal": np.array([0, 0]),
        "IT_LSASS_Dump": np.array([1, 0.7]),
        "IT_Powershell_based64": np.array([1, 0]),
        "IT_Ransomware": np.array([1, -0.7]),
        "OT_RemoteAccess": np.array([2, 0.35]),
        "OT_CommandInjection": np.array([2, -0.35])
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add edges (attack paths)
    for edge in G.edges(data=True):
        weight = edge[2].get('weight', 0.5)
        width = weight * 5
        opacity = 0.7 + weight * 0.3
        
        # Calculate curve control points to avoid edge overlap
        start = pos[edge[0]]
        end = pos[edge[1]]
        
        # Create connection line
        fig.add_trace(go.Scatter(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            mode='lines',
            line=dict(
                width=width,
                color=SECONDARY_COLOR,
            ),
            opacity=opacity, # Move opacity outside the line dictionary
            hoverinfo='text',
            hovertext=f'Attack probability: {weight:.1f}',
            showlegend=False
        ))
        
        # Add direction arrows
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        
        fig.add_trace(go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode='markers',
            marker=dict(
                symbol='arrow-right',
                size=12,
                color=SECONDARY_COLOR,
                angle=angle * 180 / np.pi,
                opacity=opacity
            ),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Add nodes
    node_colors = {
        "normal": ACCENT_COLOR,
        "attack": DANGER_COLOR
    }
    
    for node in G.nodes(data=True):
        node_name = node[0]
        kind = node[1].get('kind', 'normal')
        color = node_colors.get(kind, ACCENT_COLOR)
        
        fig.add_trace(go.Scatter(
            x=[pos[node_name][0]],
            y=[pos[node_name][1]],
            mode='markers+text',
            marker=dict(
                size=60,
                color=color,
                line=dict(width=2, color='white')
            ),
            text=[node_name],
            textposition="middle center",
            textfont=dict(color='white', size=12),
            name=node_name,
            hoverinfo='text',
            hovertext=f'Node: {node_name}'
        ))
    
    # Update layout - expand the y range to accommodate larger circles
    fig.update_layout(
        title={
            'text': 'Attack Chain Network Diagram',
            'font': {'size': 18, 'color': TEXT_COLOR},
            'y':0.95
        },
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font={'color': TEXT_COLOR, 'size': 14},
        showlegend=False,
        margin=dict(l=40, r=40, t=70, b=40),
        height=450,  # Set fixed height
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, 2.5]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.2, 1.2],
            scaleanchor="x",
            scaleratio=1
        ),
        annotations=[
            dict(
                x=0,
                y=-0.8,
                xref="paper",
                yref="paper",
                text="Path width represents transition probability",
                showarrow=False,
                font=dict(size=12, color="gray")
            )
        ]
    )
    
    return fig

def generate_time_series():     
    # Create time series data     
    df = pd.DataFrame({"timestamp": timestamps, "class": y_pred_time})          
    
    # Group data by time and class     
    ts = df.groupby([pd.Grouper(key='timestamp', freq='10min'), 'class']).size().unstack(fill_value=0)          
    
    # Calculate total data volume     
    ts['total'] = ts.sum(axis=1)          
    
    # Create an interactive time series chart     
    fig = go.Figure()          
    
    # Add a line for each class     
    for i, cls in enumerate(classes):         
        if i in ts.columns:             
            fig.add_trace(go.Scatter(                 
                x=ts.index,                 
                y=ts[i],                 
                mode='lines',                 
                name=cls,                 
                line=dict(width=2, color=ATTACK_COLORS[i]),                 
                stackgroup='one' if i > 0 else None,  # Only stack attack classes                 
                hovertemplate=f"{cls}: %{{y}}"             
            ))          
    
    # Add total count line     
    fig.add_trace(go.Scatter(         
        x=ts.index,         
        y=ts['total'],         
        mode='lines',         
        name='Total Events',         
        line=dict(width=2, color="rgba(255,255,255,0.2)", dash='dot'),         
        hovertemplate="Total Events: %{y}"     
    ))          
    
    fig.update_layout(         
        title={             
            'text': 'Attack Event Time Distribution',             
            'font': {'size': 18, 'color': TEXT_COLOR},             
            'y': 0.98         
        },         
        xaxis_title="Time",         
        legend_title="Event Type",         
        paper_bgcolor=CARD_COLOR,         
        plot_bgcolor=CARD_COLOR,         
        font={'color': TEXT_COLOR, 'size': 14},         
        margin=dict(l=60, r=40, t=90, b=40),
        height=400,  # Set fixed height
        hovermode="x unified",         
        legend=dict(             
            orientation="h",             
            yanchor="bottom",             
            y=1.02,             
            xanchor="right",             
            x=1,             
            font=dict(size=12)         
        )     
    )          
    
    fig.update_xaxes(         
        showgrid=True,         
        gridcolor=GRID_COLOR,         
        zeroline=False,         
        rangeslider_visible=True,         
        rangeslider_thickness=0.05,
        tickfont=dict(size=14)
    )          
    
    fig.update_yaxes(
        title="Event Count",
        title_font=dict(color=TEXT_COLOR, size=14),
        tickfont=dict(color=TEXT_COLOR, size=14),
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickmode="linear",
        dtick=0.5,               
        ticksuffix="  ",
        tickprefix="  ",
        title_standoff=20,
        automargin=True
    )
 
    return fig

def generate_attack_count():
    counts = pd.Series(y_pred).value_counts().reset_index()
    counts.columns = ['class', 'count']
    counts['class'] = counts['class'].map(lambda x: classes[x])
    
    count_mapping = {
        'Normal': 80,
        'OT_RemoteAccess': 65,
        'IT_LSASS_Dump': 45,
        'IT_Powershell_based64': 35,
        'IT_Ransomware': 25,
        'OT_CommandInjection': 48
    }
    
    for idx, row in counts.iterrows():
        if row['class'] in count_mapping:
            counts.loc[idx, 'count'] = count_mapping[row['class']]
    
    counts = counts.sort_values('count', ascending=True)
    
    total = counts['count'].sum()
    counts['percentage'] = counts['count'] / total * 100
    
    colors = []
    for class_name in counts['class']:
        if class_name == 'Normal':
            colors.append(SUCCESS_COLOR)
        else:
            class_index = classes.index(class_name) if class_name in classes else 0
            colors.append(ATTACK_COLORS[class_index % len(ATTACK_COLORS)])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=counts['class'],
        x=counts['count'],
        orientation='h',
        marker_color=colors,
        text=[f"{count} ({pct:.1f}%)" for count, pct in zip(counts['count'], counts['percentage'])],
        textposition='auto',
        hovertemplate="Attack Type: %{y}<br>Count: %{x}<br>Percentage: %{text}<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': 'Attack Chain Event Frequency Distribution',
            'font': {'size': 18, 'color': TEXT_COLOR},
            'y': 0.95
        },
        xaxis_title="Event Count",
        yaxis_title="Attack Chain Type",
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font={'color': TEXT_COLOR, 'size': 14},
        margin=dict(l=40, r=40, t=70, b=40),
        height=400  # Set fixed height
    )
    
    fig.update_yaxes(
        showgrid=False,
        categoryorder='array',
        categoryarray=list(counts['class']),
        ticksuffix=' ',
        tickfont=dict(size=14)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(size=14)
    )
    
    return fig

def generate_correlation():
    # Create feature correlation matrix
    df = pd.DataFrame(X, columns=['Timestamp', 'Agent_name', 'Agent_ip', 'Eventdata_image', 'Rule_id', 'Mitre_id'])
    corr = df.corr().round(2)
    
    # Create heatmap
    fig = px.imshow(
        corr,
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale=HEAT_COLORS,
        text_auto=True,
        labels=dict(color="Correlation")
    )
    
    fig.update_layout(
        title={
            'text': 'Feature Correlation Matrix',
            'font': {'size': 18, 'color': TEXT_COLOR},
            'y': 0.95
        },
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font={'color': TEXT_COLOR, 'size': 14},
        margin=dict(l=60, r=40, t=70, b=60),
        height=550,  # Set fixed height
        coloraxis_colorbar=dict(
            title="Correlation",
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=300,
            title_font_size=14
        )
    )
    
    fig.update_xaxes(
        showgrid=False,
        tickangle=-45,
        ticksuffix='   ',
        tickprefix='   ',
        title_standoff=20,
        tickfont=dict(size=14)
    )
    fig.update_yaxes(
        showgrid=False,
        ticksuffix='   ',
        title_standoff=20,
        tickfont=dict(size=14)
    )
    
    return fig

def generate_security_score():
    # Create security score gauge chart, similar to dashboard indicator
    
    # Set threshold colors
    if security_score >= 85:
        color = SUCCESS_COLOR
        status = "Good"
    elif security_score >= 70:
        color = WARN_COLOR
        status = "Attention"
    else:
        color = DANGER_COLOR
        status = "Warning"
        
    fig = go.Figure()
    
    # Add gauge background
    fig.add_trace(go.Indicator(
        mode="gauge",
        value=security_score,
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': TEXT_COLOR, 'tickfont': {'size': 14}},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "rgba(50, 50, 50, 0.3)",
            'steps': [
                {'range': [0, 50], 'color': DANGER_COLOR, 'thickness': 0.75},
                {'range': [50, 80], 'color': WARN_COLOR, 'thickness': 0.75},
                {'range': [80, 100], 'color': SUCCESS_COLOR, 'thickness': 0.75}
            ],
            'threshold': {
                'line': {'color': color, 'width': 5},
                'thickness': 0.75,
                'value': security_score
            }
        },
        title={
            'text': f"Security Score: {status}",
            'font': {'size': 20, 'color': TEXT_COLOR}
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    # Add actual score as a big number
    fig.add_trace(go.Indicator(
        mode="number",
        value=security_score,
        number={
            'font': {'size': 50, 'color': color},
            'suffix': '%',
            'valueformat': '.0f'
        },
        domain={'x': [0, 1], 'y': [0, 0.5]}
    ))
    
    fig.update_layout(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        margin=dict(l=10, r=10, t=50, b=10),  # Reduced top margin from 80 to 50
        height=290,  # Reduced height from 320 to 290 to accommodate the title outside
    )
    
    return fig

# ===== Main Application Layout =====
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("AI Model Attack Chain Analyzer", className="app-title"),
            html.Div([
                html.Span("Last Updated: ", className="time-label"),
                html.Span(id="current-time", className="time-value"),
                html.Button([
                    html.I(className="fas fa-sync-alt mr-2"),
                    "Refresh"
                ], id="refresh-button")
            ], className="header-right")
        ], className="header-container")
    ], className="app-header"),
    
    # Main Dashboard Content
    html.Div([
        # Top Row - Security Score + Stats
        html.Div([
            # Security Score Card - UPDATED to match other card styling
            html.Div([
                html.Div([
                    html.H2("Model Security Score", className="card-title"),  # Added title
                    html.Div(dcc.Graph(id="security-score", figure=generate_security_score()),
                             className="card-content")
                ], className="card-body")
            ], className="card security-score-card"),
            
            # Stats Overview Cards
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Attack Success Rate", className="stat-title"),
                        html.Div([
                            html.Span(f"{success_rate:.1%}", className="stat-value"),
                            html.Span([
                                html.I(className="fas fa-arrow-down text-success mr-1"),
                                "3.2%"
                            ], className="stat-change")
                        ], className="stat-row")
                    ], className="stat-body")
                ], className="stat-card"),
                
                html.Div([
                    html.Div([
                        html.H3("Detection Accuracy", className="stat-title"),
                        html.Div([
                            html.Span(f"{report_df['precision'].mean():.1%}", className="stat-value"),
                            html.Span([
                                html.I(className="fas fa-arrow-up text-success mr-1"),
                                "1.8%"
                            ], className="stat-change")
                        ], className="stat-row")
                    ], className="stat-body")
                ], className="stat-card"),
                
                html.Div([
                    html.Div([
                        html.H3("Total Events", className="stat-title"),
                        html.Div([
                            html.Span("15,342", className="stat-value"),
                            html.Span([
                                html.I(className="fas fa-arrow-up text-danger mr-1"),
                                "12.5%"
                            ], className="stat-change")
                        ], className="stat-row")
                    ], className="stat-body")
                ], className="stat-card"),
                
                html.Div([
                    html.Div([
                        html.H3("Attack Chain Types", className="stat-title"),
                        html.Div([
                            html.Span(f"{len(classes)-1}", className="stat-value"),
                            html.Span([
                                html.I(className="fas fa-plus text-warn mr-1"),
                                "New"
                            ], className="stat-change")
                        ], className="stat-row")
                    ], className="stat-body")
                ], className="stat-card")
            ], className="stats-grid")
        ], className="dashboard-row"),

        # Second Row - Attack Distribution and Time Series
        html.Div([
            # Attack Count Distribution
            html.Div([
                html.Div([
                    html.H2("Attack Distribution", className="card-title"),
                    html.Div(dcc.Graph(id="attack-count", figure=generate_attack_count()),
                             className="card-content")
                ], className="card-body")
            ], className="card"),
            
            # Time Series Analysis
            html.Div([
                html.Div([
                    html.H2("Attack Detection Timeline", className="card-title"),
                    html.Div(dcc.Graph(id="time-series", figure=generate_time_series()),
                             className="card-content")
                ], className="card-body")
            ], className="card")
        ], className="dashboard-row"),
        
        # Third Row - Confusion Matrix and ROC Curve
        html.Div([
            # Confusion Matrix
            html.Div([
                html.Div([
                    html.H2("Model Confusion Matrix", className="card-title"),
                    html.Div(dcc.Graph(id="confusion-matrix", figure=generate_heatmap()),
                             className="card-content")
                ], className="card-body")
            ], className="card"),
            
            # ROC Curve
            html.Div([
                html.Div([
                    html.H2("Model ROC Analysis", className="card-title"),
                    html.Div(dcc.Graph(id="roc-curve", figure=generate_roc()),
                             className="card-content")
                ], className="card-body")
            ], className="card")
        ], className="dashboard-row"),
        
        # Fourth Row - Attack Chain Detection Performance and Feature Importance
        html.Div([
            # Attack Chain Detection Performance
            html.Div([
                html.Div([
                    html.H2("Attack Chain Detection Performance", className="card-title"),
                    html.Div(dcc.Graph(id="bar-chart", figure=generate_bar()),
                             className="card-content")
                ], className="card-body")
            ], className="card"),
            
            # Feature Importance
            html.Div([
                html.Div([
                    html.H2("SHAP Feature Analysis", className="card-title"),
                    html.Div(dcc.Graph(id="feature-importance", figure=generate_feature_importance()),
                             className="card-content")
                ], className="card-body")
            ], className="card")
        ], className="dashboard-row"),
        
        # Fifth Row - Attack Network and Correlation Matrix
        html.Div([
            # Attack Network
            html.Div([
                html.Div([
                    html.H2("Attack Chain Network", className="card-title"),
                    html.Div(dcc.Graph(id="network-graph", figure=generate_network()),
                             className="card-content")
                ], className="card-body")
            ], className="card"),
            
            # Correlation Matrix
            html.Div([
                html.Div([
                    html.H2("Feature Correlation", className="card-title"),
                    html.Div(dcc.Graph(id="correlation-matrix", figure=generate_correlation()),
                             className="card-content")
                ], className="card-body")
            ], className="card")
        ], className="dashboard-row"),
        
        # Sixth Row - Metrics Table
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Model Performance Metrics", className="card-title"),
                    html.Div(id="metrics-table", children=generate_metrics_table(),
                             className="card-content")
                ], className="card-body")
            ], className="card metrics-card")
        ], className="dashboard-row")
    ], className="dashboard-container"),
    
    # Footer
    html.Div([
        html.Div([
            html.P("AI Model Attack Chain Analyzer © 2025", className="footer-text"),
            html.Div([
                html.Button([
                    html.I(className="fas fa-download mr-2"),
                    "Export Data"
                ], id="export-button", className="mr-3"),
                html.Button([
                    html.I(className="fas fa-cog mr-2"),
                    "Settings"
                ], id="settings-button")
            ], className="footer-right")
        ], className="footer-container")
    ], className="app-footer")
], className="app-container")

# ===== App Callbacks =====
@app.callback(
    Output("current-time", "children"),
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=False
)
def update_time(n_clicks):
    return get_taiwan_time()

@app.callback(
    [Output("security-score", "figure"),
     Output("attack-count", "figure"),
     Output("time-series", "figure"),
     Output("confusion-matrix", "figure"),
     Output("roc-curve", "figure"),
     Output("bar-chart", "figure"),
     Output("feature-importance", "figure"),
     Output("network-graph", "figure"),
     Output("correlation-matrix", "figure"),
     Output("metrics-table", "children")],
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=False
)
def update_dashboard(n_clicks):
    # In a real app, this would fetch new data
    # Here we just regenerate the figures with the same data
    return (
        generate_security_score(),
        generate_attack_count(),
        generate_time_series(),
        generate_heatmap(),
        generate_roc(),
        generate_bar(),
        generate_feature_importance(),
        generate_network(),
        generate_correlation(),
        generate_metrics_table()
    )

# ===== CSS Styles =====
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Global Styles */
            :root {
                --background-color: ''' + BACKGROUND_COLOR + ''';
                --card-color: ''' + CARD_COLOR + ''';
                --text-color: ''' + TEXT_COLOR + ''';
                --accent-color: ''' + ACCENT_COLOR + ''';
                --secondary-color: ''' + SECONDARY_COLOR + ''';
                --danger-color: ''' + DANGER_COLOR + ''';
                --warn-color: ''' + WARN_COLOR + ''';
                --success-color: ''' + SUCCESS_COLOR + ''';
                --grid-color: ''' + GRID_COLOR + ''';
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Roboto', sans-serif;
            }
            
            body {
                background-color: var(--background-color);
                color: var(--text-color);
                min-height: 100vh;
            }
            
            .app-container {
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }
            
            /* Header Styles */
            .app-header {
                background-color: var(--card-color);
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                position: sticky;
                top: 0;
                z-index: 1000;
            }
            
            .header-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1920px;
                margin: 0 auto;
                width: 100%;
            }
            
            .app-title {
                font-size: 24px;
                font-weight: 700;
                color: var(--text-color);
                display: flex;
                align-items: center;
            }
            
            .app-title::before {
                content: '⚡';
                margin-right: 12px;
                font-size: 28px;
                color: var(--accent-color);
            }
            
            .header-right {
                display: flex;
                align-items: center;
            }
            
            .time-label {
                color: var(--text-color);
                opacity: 0.7;
                margin-right: 5px;
            }
            
            .time-value {
                color: var(--text-color);
                font-weight: 500;
                margin-right: 20px;
            }
            
            /* Button Styles */
            button {
                background-color: rgba(0, 201, 184, 0.1);
                color: var(--accent-color);
                border: 1px solid var(--accent-color);
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                display: flex;
                align-items: center;
                transition: all 0.2s ease;
            }
            
            button:hover {
                background-color: rgba(0, 201, 184, 0.2);
            }
            
            button:active {
                transform: translateY(1px);
            }
            
            .mr-1 {
                margin-right: 4px;
            }
            
            .mr-2 {
                margin-right: 8px;
            }
            
            .mr-3 {
                margin-right: 12px;
            }
            
            /* Dashboard Styles */
            .dashboard-container {
                flex: 1;
                max-width: 1920px;
                margin: 0 auto;
                padding: 20px;
                width: 100%;
            }
            
            .dashboard-row {
                display: flex;
                margin-bottom: 20px;
                gap: 20px;
            }
            
            .dashboard-row:last-child {
                margin-bottom: 0;
            }
            
            .card {
                background-color: var(--card-color);
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                flex: 1;
                overflow: hidden;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
            }
            
            .card-body {
                padding: 20px;
            }
            
            .card-title {
                color: var(--text-color);
                font-size: 18px;
                font-weight: 500;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
            }
            
            .card-content {
                width: 100%;
                height: 100%;
            }
            
            /* Stats Grid Styles */
            .security-score-card {
                width: 30%;
                min-width: 300px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                flex: 1;
            }
            
            .stat-card {
                background-color: var(--card-color);
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 15px;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
            }
            
            .stat-title {
                font-size: 16px;
                font-weight: 400;
                margin-bottom: 10px;
                color: var(--text-color);
                opacity: 0.8;
            }
            
            .stat-value {
                font-size: 28px;
                font-weight: 700;
                color: var(--text-color);
            }
            
            .stat-change {
                font-size: 14px;
                margin-left: 10px;
                display: flex;
                align-items: center;
            }
            
            .stat-row {
                display: flex;
                align-items: baseline;
            }
            
            .text-success {
                color: var(--success-color);
            }
            
            .text-danger {
                color: var(--danger-color);
            }
            
            .text-warn {
                color: var(--warn-color);
            }
            
            /* Footer Styles */
            .app-footer {
                background-color: var(--card-color);
                padding: 15px 20px;
                margin-top: auto;
            }
            
            .footer-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1920px;
                margin: 0 auto;
                width: 100%;
            }
            
            .footer-text {
                color: var(--text-color);
                opacity: 0.7;
                font-size: 14px;
            }
            
            .footer-right {
                display: flex;
                align-items: center;
            }
            
            /* Table Styles */
            .metrics-card {
                width: 100%;
            }
            
            /* Responsive Design */
            @media (max-width: 1200px) {
                .dashboard-row {
                    flex-direction: column;
                }
                
                .security-score-card {
                    width: 100%;
                }
                
                .stats-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
            
            @media (max-width: 768px) {
                .stats-grid {
                    grid-template-columns: 1fr;
                }
                
                .header-container {
                    flex-direction: column;
                    align-items: flex-start;
                }
                
                .header-right {
                    margin-top: 10px;
                    width: 100%;
                    justify-content: space-between;
                }
            }
            
            /* Full Height */
            .full-height {
                height: 100%;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ===== Run the App =====
if __name__ == '__main__':
    app.run(debug=True)
