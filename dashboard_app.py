"""
Cardiovascular Disease Analysis Dashboard - IMPROVED VERSION
Interactive Plotly Dash application with brushing & linking
Now with 5 visualizations, reset button, and better UX
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
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
            html.H1("Cardiovascular Disease Analysis Dashboard",
                   className="text-center mb-3 mt-4",
                   style={'color': '#2c3e50', 'fontWeight': 'bold'}),
            html.P("Interactive exploration of cardiovascular disease patterns. "
                  "Click feature bars or drag to select age range in scatter plot.",
                  className="text-center text-muted mb-4",
                  style={'fontSize': '16px'}),
        ])
    ]),

    # Info Card with Reset Button
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Current Selection", className="mb-2", style={'fontWeight': 'bold'}),
                            html.Div(id='selection-info',
                                    children="No filters applied - showing all data",
                                    style={'fontSize': '14px', 'color': '#555'})
                        ], width=9),
                        dbc.Col([
                            dbc.Button("Reset All", id='reset-button', color="danger",
                                     size="sm", className="float-end",
                                     style={'fontWeight': 'bold'})
                        ], width=3)
                    ])
                ])
            ], color="light", className="mb-4", style={'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ])
    ]),

    # Store components for sharing state between callbacks
    dcc.Store(id='selected-feature'),
    dcc.Store(id='selected-age-range'),
    dcc.Store(id='reset-trigger', data=0),

    # Main Dashboard Grid - 5 charts layout
    # Top Row: 3 columns
    dbc.Row([
        # Chart 1: Feature Importance
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H6("Feature Importance for Disease Prediction",
                           className="mb-0", style={'fontWeight': 'bold'}),
                    style={'backgroundColor': '#2c3e50', 'color': 'white'}
                ),
                dbc.CardBody([
                    html.P("Click any bar to see its distribution →",
                          className="text-muted small mb-2"),
                    dcc.Graph(id='feature-importance-chart',
                             config={'displayModeBar': False},
                             style={'height': '350px'})
                ])
            ], className="h-100", style={'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'})
        ], width=4, className="mb-4"),

        # Chart 2: Disease Prevalence
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H6(" Disease Prevalence by Risk Factors",
                           className="mb-0", style={'fontWeight': 'bold'}),
                    style={'backgroundColor': '#2c3e50', 'color': 'white'}
                ),
                dbc.CardBody([
                    html.P("Updates when age range is selected →",
                          className="text-muted small mb-2"),
                    dcc.Graph(id='disease-prevalence-chart',
                             config={'displayModeBar': False},
                             style={'height': '350px'})
                ])
            ], className="h-100", style={'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'})
        ], width=4, className="mb-4"),

        # Chart 3: Health Risk Score (NEW!)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H6(" Health Risk Score Distribution",
                           className="mb-0", style={'fontWeight': 'bold'}),
                    style={'backgroundColor': '#2c3e50', 'color': 'white'}
                ),
                dbc.CardBody([
                    html.P("Shows composite risk levels →",
                          className="text-muted small mb-2"),
                    dcc.Graph(id='risk-score-chart',
                             config={'displayModeBar': False},
                             style={'height': '350px'})
                ])
            ], className="h-100", style={'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'})
        ], width=4, className="mb-4"),
    ]),

    # Bottom Row: 2 columns
    dbc.Row([
        # Chart 4: Scatter Plot
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H6("Age vs Blood Pressure - Interactive Selection",
                           className="mb-0", style={'fontWeight': 'bold'}),
                    style={'backgroundColor': '#2c3e50', 'color': 'white'}
                ),
                dbc.CardBody([
                    dbc.Alert([
                        html.Strong("How to use: "),
                        "1) Click the 'Box Select' tool in the chart toolbar above. ",
                        "2) Drag to select an age range. ",
                        "3) Watch all other charts update!"
                    ], color="info", className="py-2 px-3 mb-2 small"),
                    dcc.Graph(id='scatter-plot-chart',
                             config={'displayModeBar': True,
                                    'modeBarButtonsToRemove': ['lasso2d', 'autoScale2d'],
                                    'displaylogo': False},
                             style={'height': '400px'})
                ])
            ], className="h-100", style={'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'})
        ], width=6, className="mb-4"),

        # Chart 5: Correlation/Distribution
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H6("Feature Analysis",
                           className="mb-0", style={'fontWeight': 'bold'}),
                    style={'backgroundColor': '#2c3e50', 'color': 'white'}
                ),
                dbc.CardBody([
                    html.P("Shows correlation matrix or selected feature distribution",
                          className="text-muted small mb-2"),
                    dcc.Graph(id='correlation-chart',
                             config={'displayModeBar': False},
                             style={'height': '400px'})
                ])
            ], className="h-100", style={'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'})
        ], width=6, className="mb-4"),
    ]),

    # Footer with stats
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='stats-footer',
                            className="text-center",
                            style={'fontSize': '14px', 'fontWeight': '500'})
                ])
            ], color="dark", inverse=True, style={'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ])
    ])

], fluid=True, style={'backgroundColor': '#ecf0f1', 'minHeight': '100vh', 'paddingBottom': '20px'})

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
    colors = ['#e74c3c' if feat == selected_feature else '#3498db'
             for feat in feature_importance_df['Feature']]

    fig.add_trace(go.Bar(
        x=feature_importance_df['Importance'],
        y=feature_importance_df['Feature'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='#2c3e50', width=1)),
        text=feature_importance_df['Importance'].round(3),
        textposition='auto',
        textfont=dict(size=10, color='white', family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<br><i>Click to explore</i><extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        hovermode='closest',
        font=dict(size=11)
    )

    fig.update_xaxes(showgrid=True, gridcolor='#e0e0e0', gridwidth=1)
    fig.update_yaxes(tickfont=dict(size=10))

    return fig

# ============================================================================
# CALLBACK 2: Disease Prevalence by Demographics
# ============================================================================

@app.callback(
    Output('disease-prevalence-chart', 'figure'),
    [Input('selected-age-range', 'data')]
)
def update_disease_prevalence(age_range):
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
    chol_map = {1: 'Normal', 2: 'Above', 3: 'High'}
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
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        hovermode='closest',
        font=dict(size=11)
    )

    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Prevalence: %{y:.1f}%<br>Patients: %{customdata[0]:,}<extra></extra>'
    )

    fig.update_xaxes(showgrid=False, tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor='#e0e0e0', range=[0, 100])

    return fig

# ============================================================================
# CALLBACK 3: Health Risk Score Distribution (NEW!)
# ============================================================================

@app.callback(
    Output('risk-score-chart', 'figure'),
    [Input('selected-age-range', 'data')]
)
def update_risk_score(age_range):
    """Update health risk score distribution with disease overlay."""

    # Filter data
    filtered_df = df.copy()
    if age_range:
        filtered_df = filtered_df[
            (filtered_df['age_years'] >= age_range[0]) &
            (filtered_df['age_years'] <= age_range[1])
        ]

    # Calculate disease prevalence by risk score
    risk_analysis = filtered_df.groupby('health_risk_score').agg({
        'cardio': ['count', 'mean']
    }).reset_index()
    risk_analysis.columns = ['Risk_Score', 'Count', 'Disease_Rate']
    risk_analysis['Disease_Rate'] *= 100

    # Create dual-axis chart
    fig = go.Figure()

    # Bar chart: Count
    fig.add_trace(go.Bar(
        x=risk_analysis['Risk_Score'],
        y=risk_analysis['Count'],
        name='Patient Count',
        marker_color='#3498db',
        yaxis='y',
        hovertemplate='Risk Score: %{x}<br>Patients: %{y:,}<extra></extra>'
    ))

    # Line chart: Disease Rate
    fig.add_trace(go.Scatter(
        x=risk_analysis['Risk_Score'],
        y=risk_analysis['Disease_Rate'],
        name='Disease Rate',
        mode='lines+markers',
        marker=dict(size=8, color='#e74c3c', line=dict(width=2, color='white')),
        line=dict(width=3, color='#e74c3c'),
        yaxis='y2',
        hovertemplate='Risk Score: %{x}<br>Disease Rate: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Health Risk Score (0=Low, 12=High)",
        yaxis_title="Number of Patients",
        yaxis2=dict(
            title="Disease Prevalence (%)",
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        height=350,
        margin=dict(l=10, r=50, t=10, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        hovermode='x unified',
        font=dict(size=11)
    )

    fig.update_xaxes(showgrid=True, gridcolor='#e0e0e0', dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor='#e0e0e0')

    return fig

# ============================================================================
# CALLBACK 4: Scatter Plot with Brushing (IMPROVED)
# ============================================================================

@app.callback(
    Output('scatter-plot-chart', 'figure'),
    [Input('selected-age-range', 'data')]
)
def update_scatter_plot(age_range):
    """Update scatter plot with improved UX."""

    # IMPROVED: Sample only 2000 points for clarity
    plot_df = df.sample(min(2000, len(df)), random_state=42).copy()

    # Map disease values to clear labels
    plot_df['Disease_Status'] = plot_df['cardio'].map({
        0: 'Healthy',
        1: 'Has Disease'
    })

    # Create scatter plot with IMPROVED opacity
    fig = px.scatter(
        plot_df,
        x='age_years',
        y='ap_hi',
        color='Disease_Status',
        opacity=0.7,  # IMPROVED: Increased from 0.6 to 0.7
        color_discrete_map={'Healthy': '#2ecc71', 'Has Disease': '#e74c3c'},
        labels={
            'age_years': 'Age (years)',
            'ap_hi': 'Systolic Blood Pressure (mmHg)',
            'Disease_Status': 'Patient Status'
        },
        hover_data={
            'age_years': True,
            'ap_hi': True,
            'Disease_Status': False,
            'ap_lo': ':.0f',
            'cholesterol': True,
            'weight': ':.0f'
        }
    )

    # Highlight selected range
    if age_range:
        # Add shaded region for selected range
        fig.add_vrect(
            x0=age_range[0], x1=age_range[1],
            fillcolor="#3498db", opacity=0.1,
            layer="below", line_width=0,
            annotation_text=f"Selected: {age_range[0]:.0f}-{age_range[1]:.0f} years",
            annotation_position="top left"
        )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        legend=dict(
            title=dict(text="<b>Patient Status</b>", font=dict(size=12)),
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#2c3e50',
            borderwidth=1
        ),
        dragmode='select',  # Enable box select
        font=dict(size=11)
    )

    # IMPROVED: Larger markers
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='white')))

    fig.update_xaxes(showgrid=True, gridcolor='#e0e0e0', gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor='#e0e0e0', gridwidth=1)

    return fig

# ============================================================================
# CALLBACK 5: Correlation/Distribution Chart
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

    # Map disease for clear labels
    filtered_df['Disease_Status'] = filtered_df['cardio'].map({
        0: 'Healthy',
        1: 'Has Disease'
    })

    if selected_feature and selected_feature in filtered_df.columns:
        # Show distribution of selected feature
        fig = go.Figure()

        # Distribution for Healthy
        fig.add_trace(go.Histogram(
            x=filtered_df[filtered_df['Disease_Status'] == 'Healthy'][selected_feature],
            name='Healthy',
            opacity=0.75,
            marker_color='#2ecc71',
            marker_line=dict(color='white', width=1),
            nbinsx=30
        ))

        # Distribution for Has Disease
        fig.add_trace(go.Histogram(
            x=filtered_df[filtered_df['Disease_Status'] == 'Has Disease'][selected_feature],
            name='Has Disease',
            opacity=0.75,
            marker_color='#e74c3c',
            marker_line=dict(color='white', width=1),
            nbinsx=30
        ))

        fig.update_layout(
            title=dict(text=f"<b>{selected_feature}</b> Distribution by Disease Status",
                      font=dict(size=13)),
            xaxis_title=selected_feature,
            yaxis_title="Number of Patients",
            barmode='overlay',
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            font=dict(size=11)
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
            textfont={"size": 10, "color": "white"},
            colorbar=dict(title="Correlation"),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text="<b>Feature Correlation Matrix</b>", font=dict(size=13)),
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='white',
            xaxis=dict(side='bottom'),
            font=dict(size=11)
        )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig

# ============================================================================
# CALLBACK 6: Brushing & Linking - Feature Click
# ============================================================================

@app.callback(
    Output('selected-feature', 'data'),
    [Input('feature-importance-chart', 'clickData'),
     Input('reset-button', 'n_clicks')],
    [State('selected-feature', 'data')]
)
def feature_click(clickData, reset_clicks, current_feature):
    """Handle feature importance bar click - BRUSHING."""

    ctx = callback_context
    if not ctx.triggered:
        return None

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-button':
        return None

    if clickData is None:
        return current_feature

    # Extract clicked feature
    feature = clickData['points'][0]['y']
    return feature

# ============================================================================
# CALLBACK 7: Brushing & Linking - Scatter Selection
# ============================================================================

@app.callback(
    Output('selected-age-range', 'data'),
    [Input('scatter-plot-chart', 'selectedData'),
     Input('reset-button', 'n_clicks')],
    [State('selected-age-range', 'data')]
)
def scatter_selection(selectedData, reset_clicks, current_range):
    """Handle scatter plot box/lasso selection - BRUSHING."""

    ctx = callback_context
    if not ctx.triggered:
        return None

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-button':
        return None

    if selectedData is None or len(selectedData['points']) == 0:
        return current_range

    # Get age range from selected points
    ages = [point['x'] for point in selectedData['points']]
    age_range = [min(ages), max(ages)]

    return age_range

# ============================================================================
# CALLBACK 8: Update Selection Info
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
        info_parts.append(html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#2ecc71', 'marginRight': '8px'}),
            html.Strong("Feature Selected: "),
            html.Span(selected_feature, style={'color': '#e74c3c', 'fontWeight': 'bold'}),
            html.Span(" (click another bar to change)", className="text-muted small ms-2")
        ], className="mb-1"))

    if age_range:
        info_parts.append(html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#2ecc71', 'marginRight': '8px'}),
            html.Strong("Age Range: "),
            html.Span(f"{age_range[0]:.0f} - {age_range[1]:.0f} years",
                     style={'color': '#3498db', 'fontWeight': 'bold'}),
            html.Span(" (drag new box to change)", className="text-muted small ms-2")
        ], className="mb-1"))

    if not info_parts:
        return html.Div([
            html.I(className="fas fa-info-circle", style={'color': '#3498db', 'marginRight': '8px'}),
            "No filters applied - Click feature bars or drag box in scatter plot to filter data"
        ])

    return html.Div(info_parts)

# ============================================================================
# CALLBACK 9: Update Stats Footer
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
    disease_count = int(filtered_df['cardio'].sum())
    healthy_count = total - disease_count
    disease_pct = (disease_count / total * 100) if total > 0 else 0

    return html.Div([
        html.Span([
            html.Strong("Showing: "),
            f"{total:,} patients "
        ]),
        html.Span("| ", className="mx-2"),
        html.Span([
            html.Strong("Healthy: "),
            html.Span(f"{healthy_count:,} ({100-disease_pct:.1f}%)", style={'color': '#2ecc71'})
        ]),
        html.Span(" | ", className="mx-2"),
        html.Span([
            html.Strong("Has Disease: "),
            html.Span(f"{disease_count:,} ({disease_pct:.1f}%)", style={'color': '#e74c3c'})
        ]),
        html.Span(" | ", className="mx-2"),
        html.Span([
            html.Strong("Total Dataset: "),
            f"{len(df):,} records"
        ])
    ])

# ============================================================================
# RUN SERVER
# ============================================================================

# Expose server for deployment (Render, Heroku, etc.)
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)