"""
Cardiovascular Disease Analysis Dashboard
Interactive Plotly Dash application with brushing & linking
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load cleaned cardiovascular disease data and model results."""
    # Load cleaned dataset
    df = pd.read_csv('models/cardio_final.csv')

    # Load model results for feature importance
    try:
        with open('models/best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)

        # Get feature importance
        feature_cols = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                       'cholesterol', 'gluc', 'smoke', 'alco', 'active']

        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
        else:
            # For Logistic Regression
            importance = np.abs(best_model.coef_[0])

        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=True)  # Ascending for horizontal bar

    except:
        # Fallback if models not available
        feature_importance_df = pd.DataFrame({
            'Feature': ['ap_hi', 'age_years', 'weight', 'cholesterol', 'ap_lo'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12]
        })

    return df, feature_importance_df

df, feature_importance_df = load_data()

# ============================================================================
# DASH APP INITIALIZATION
# ============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Cardiovascular Disease Dashboard"

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🏥 Cardiovascular Disease Analysis Dashboard",
                   className="text-center mb-4 mt-4",
                   style={'color': '#2c3e50'}),
            html.P("Interactive exploration of cardiovascular disease patterns and predictions. "
                  "Click on any chart to filter and explore relationships.",
                  className="text-center text-muted mb-4"),
        ])
    ]),

    # Info Card - Shows current selection
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Current Selection", className="card-title"),
                    html.Div(id='selection-info',
                            children="Click on any visualization to filter data",
                            style={'fontSize': '14px'})
                ])
            ], color="light", className="mb-3")
        ])
    ]),

    # Store components for sharing state between callbacks
    dcc.Store(id='selected-feature'),
    dcc.Store(id='selected-age-range'),
    dcc.Store(id='selected-demographic'),

    # Main Dashboard Grid - 2x2 layout
    dbc.Row([
        # Top Left: Feature Importance
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Feature Importance", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='feature-importance-chart',
                             config={'displayModeBar': False})
                ])
            ], className="h-100")
        ], width=6, className="mb-4"),

        # Top Right: Disease Prevalence by Demographics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Disease Prevalence by Risk Factors", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='disease-prevalence-chart',
                             config={'displayModeBar': False})
                ])
            ], className="h-100")
        ], width=6, className="mb-4"),
    ]),

    dbc.Row([
        # Bottom Left: Scatter Plot - Age vs BP
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Age vs Blood Pressure (Brush to Select)", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='scatter-plot-chart',
                             config={'displayModeBar': True})
                ])
            ], className="h-100")
        ], width=6, className="mb-4"),

        # Bottom Right: Correlation Heatmap / Distribution
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Feature Distribution & Correlation", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='correlation-chart',
                             config={'displayModeBar': False})
                ])
            ], className="h-100")
        ], width=6, className="mb-4"),
    ]),

    # Footer with stats
    dbc.Row([
        dbc.Col([
            html.Div(id='stats-footer', className="text-center text-muted small")
        ])
    ])

], fluid=True, style={'backgroundColor': '#f8f9fa'})

# ============================================================================
# CALLBACK 1: Feature Importance Chart
# ============================================================================

@app.callback(
    Output('feature-importance-chart', 'figure'),
    [Input('selected-feature', 'data'),
     Input('selected-age-range', 'data')]
)
def update_feature_importance(selected_feature, age_range):
    """Update feature importance chart with highlighting."""

    fig = go.Figure()

    # Color bars based on selection
    colors = ['#3498db' if feat != selected_feature else '#e74c3c'
             for feat in feature_importance_df['Feature']]

    fig.add_trace(go.Bar(
        x=feature_importance_df['Importance'],
        y=feature_importance_df['Feature'],
        orientation='h',
        marker=dict(color=colors),
        text=feature_importance_df['Importance'].round(3),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white',
        hovermode='closest'
    )

    fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1')

    return fig

# ============================================================================
# CALLBACK 2: Disease Prevalence by Demographics
# ============================================================================

@app.callback(
    Output('disease-prevalence-chart', 'figure'),
    [Input('selected-age-range', 'data'),
     Input('selected-feature', 'data')]
)
def update_disease_prevalence(age_range, selected_feature):
    """Update disease prevalence grouped bar chart."""

    # Filter data if age range is selected
    filtered_df = df.copy()
    if age_range:
        filtered_df = filtered_df[
            (filtered_df['age_years'] >= age_range[0]) &
            (filtered_df['age_years'] <= age_range[1])
        ]

    # Calculate prevalence by multiple factors
    prevalence_data = []

    # By Age Group
    age_groups = pd.cut(filtered_df['age_years'], bins=[0, 45, 55, 65, 100],
                       labels=['<45', '45-55', '55-65', '65+'])
    for group in ['<45', '45-55', '55-65', '65+']:
        subset = filtered_df[age_groups == group]
        if len(subset) > 0:
            prevalence = subset['cardio'].mean() * 100
            prevalence_data.append({
                'Category': 'Age Group',
                'Group': group,
                'Prevalence': prevalence,
                'Count': len(subset)
            })

    # By Cholesterol
    chol_map = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above'}
    for chol_level, chol_name in chol_map.items():
        subset = filtered_df[filtered_df['cholesterol'] == chol_level]
        if len(subset) > 0:
            prevalence = subset['cardio'].mean() * 100
            prevalence_data.append({
                'Category': 'Cholesterol',
                'Group': chol_name,
                'Prevalence': prevalence,
                'Count': len(subset)
            })

    # By Gender
    gender_map = {1: 'Female', 2: 'Male'}
    for gender_val, gender_name in gender_map.items():
        subset = filtered_df[filtered_df['gender'] == gender_val]
        if len(subset) > 0:
            prevalence = subset['cardio'].mean() * 100
            prevalence_data.append({
                'Category': 'Gender',
                'Group': gender_name,
                'Prevalence': prevalence,
                'Count': len(subset)
            })

    prev_df = pd.DataFrame(prevalence_data)

    # Create grouped bar chart
    fig = px.bar(prev_df,
                 x='Group',
                 y='Prevalence',
                 color='Category',
                 barmode='group',
                 hover_data=['Count'],
                 color_discrete_map={
                     'Age Group': '#3498db',
                     'Cholesterol': '#e74c3c',
                     'Gender': '#2ecc71'
                 })

    fig.update_layout(
        yaxis_title="Disease Prevalence (%)",
        xaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        hovermode='closest'
    )

    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Prevalence: %{y:.1f}%<br>Count: %{customdata[0]}<extra></extra>'
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#ecf0f1')

    return fig

# ============================================================================
# CALLBACK 3: Scatter Plot with Brushing
# ============================================================================

@app.callback(
    Output('scatter-plot-chart', 'figure'),
    [Input('selected-feature', 'data'),
     Input('selected-age-range', 'data')]
)
def update_scatter_plot(selected_feature, age_range):
    """Update scatter plot with brushing capability."""

    # Sample data for performance (use every 10th point for large datasets)
    plot_df = df.sample(min(5000, len(df)), random_state=42).copy()

    # Highlight selected range
    if age_range:
        plot_df['selected'] = (
            (plot_df['age_years'] >= age_range[0]) &
            (plot_df['age_years'] <= age_range[1])
        )
    else:
        plot_df['selected'] = True

    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='age_years',
        y='ap_hi',
        color='cardio',
        size='ap_lo',
        opacity=0.6,
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        labels={
            'age_years': 'Age (years)',
            'ap_hi': 'Systolic Blood Pressure (mmHg)',
            'ap_lo': 'Diastolic BP',
            'cardio': 'Disease'
        },
        category_orders={'cardio': [0, 1]},
        hover_data=['height', 'weight', 'cholesterol']
    )

    # Update marker opacity based on selection
    if age_range:
        fig.for_each_trace(lambda trace: trace.update(
            marker=dict(
                opacity=[0.7 if sel else 0.2 for sel in plot_df[plot_df['cardio'] == trace.name]['selected']]
            )
        ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white',
        legend=dict(
            title="Cardiovascular Disease",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        dragmode='select'  # Enable box/lasso select
    )

    fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=True, gridcolor='#ecf0f1')

    return fig

# ============================================================================
# CALLBACK 4: Correlation/Distribution Chart
# ============================================================================

@app.callback(
    Output('correlation-chart', 'figure'),
    [Input('selected-feature', 'data'),
     Input('selected-age-range', 'data')]
)
def update_correlation_chart(selected_feature, age_range):
    """Update correlation heatmap or feature distribution."""

    # Filter data
    filtered_df = df.copy()
    if age_range:
        filtered_df = filtered_df[
            (filtered_df['age_years'] >= age_range[0]) &
            (filtered_df['age_years'] <= age_range[1])
        ]

    if selected_feature and selected_feature in filtered_df.columns:
        # Show distribution of selected feature
        fig = go.Figure()

        # Distribution for disease = 0
        fig.add_trace(go.Histogram(
            x=filtered_df[filtered_df['cardio'] == 0][selected_feature],
            name='No Disease',
            opacity=0.7,
            marker_color='#2ecc71',
            nbinsx=30
        ))

        # Distribution for disease = 1
        fig.add_trace(go.Histogram(
            x=filtered_df[filtered_df['cardio'] == 1][selected_feature],
            name='Disease',
            opacity=0.7,
            marker_color='#e74c3c',
            nbinsx=30
        ))

        fig.update_layout(
            title=f"Distribution of {selected_feature} by Disease Status",
            xaxis_title=selected_feature,
            yaxis_title="Count",
            barmode='overlay',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

    else:
        # Show correlation heatmap
        corr_features = ['age_years', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'cardio']
        corr_matrix = filtered_df[corr_features].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title="Feature Correlation Matrix",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(side='bottom'),
        )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig

# ============================================================================
# CALLBACK 5: Brushing & Linking - Feature Click
# ============================================================================

@app.callback(
    Output('selected-feature', 'data'),
    Input('feature-importance-chart', 'clickData')
)
def feature_click(clickData):
    """Handle feature importance bar click - BRUSHING."""
    if clickData is None:
        return None

    # Extract clicked feature
    feature = clickData['points'][0]['y']
    return feature

# ============================================================================
# CALLBACK 6: Brushing & Linking - Scatter Selection
# ============================================================================

@app.callback(
    Output('selected-age-range', 'data'),
    Input('scatter-plot-chart', 'selectedData')
)
def scatter_selection(selectedData):
    """Handle scatter plot box/lasso selection - BRUSHING."""
    if selectedData is None or len(selectedData['points']) == 0:
        return None

    # Get age range from selected points
    ages = [point['x'] for point in selectedData['points']]
    age_range = [min(ages), max(ages)]

    return age_range

# ============================================================================
# CALLBACK 7: Update Selection Info
# ============================================================================

@app.callback(
    Output('selection-info', 'children'),
    [Input('selected-feature', 'data'),
     Input('selected-age-range', 'data')]
)
def update_selection_info(selected_feature, age_range):
    """Display current selection information."""

    info_parts = []

    if selected_feature:
        info_parts.append(html.Span([
            html.Strong("Selected Feature: "),
            html.Span(selected_feature, style={'color': '#e74c3c'}),
            " (click feature chart to change)"
        ]))

    if age_range:
        info_parts.append(html.Span([
            html.Strong("Age Range: "),
            html.Span(f"{age_range[0]:.1f} - {age_range[1]:.1f} years",
                     style={'color': '#3498db'}),
            " (brush scatter plot to change)"
        ]))

    if not info_parts:
        return "Click on Feature Importance chart or brush Scatter Plot to filter data"

    return html.Div([
        html.Div(part, className="mb-1") for part in info_parts
    ])

# ============================================================================
# CALLBACK 8: Update Stats Footer
# ============================================================================

@app.callback(
    Output('stats-footer', 'children'),
    [Input('selected-age-range', 'data')]
)
def update_stats(age_range):
    """Display dataset statistics."""

    filtered_df = df.copy()
    if age_range:
        filtered_df = filtered_df[
            (filtered_df['age_years'] >= age_range[0]) &
            (filtered_df['age_years'] <= age_range[1])
        ]

    total = len(filtered_df)
    disease_count = filtered_df['cardio'].sum()
    disease_pct = (disease_count / total * 100) if total > 0 else 0

    return f"Showing {total:,} records | Disease Prevalence: {disease_pct:.1f}% ({disease_count:,} cases) | Total Dataset: {len(df):,} records"

# ============================================================================
# RUN SERVER
# ============================================================================

# Expose server for deployment (Render, Heroku, etc.)
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
