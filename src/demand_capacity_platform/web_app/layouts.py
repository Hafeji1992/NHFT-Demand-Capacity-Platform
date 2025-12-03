"""
Dash Application Layouts Module.

Defines the layout components for the web application.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_navbar() -> dbc.NavbarSimple:
    """Create the navigation bar."""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("Forecasting", href="/forecasting")),
            dbc.NavItem(dbc.NavLink("Capacity", href="/capacity")),
            dbc.NavItem(dbc.NavLink("Simulation", href="/simulation")),
        ],
        brand="NHS Demand & Capacity Platform",
        brand_href="/",
        color="primary",
        dark=True,
        fluid=True,
    )


def create_sidebar() -> dbc.Card:
    """Create the sidebar with controls."""
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Controls", className="mb-0")),
            dbc.CardBody(
                [
                    # Service selection
                    html.Label("Select Service:", className="fw-bold"),
                    dcc.Dropdown(
                        id="service-dropdown",
                        options=[
                            {"label": "Adult Mental Health", "value": "Adult Mental Health"},
                            {"label": "CAMHS", "value": "CAMHS"},
                            {"label": "Community Nursing", "value": "Community Nursing"},
                            {"label": "Physiotherapy", "value": "Physiotherapy"},
                            {"label": "Occupational Therapy", "value": "Occupational Therapy"},
                        ],
                        value="Adult Mental Health",
                        clearable=False,
                        className="mb-3",
                    ),

                    html.Hr(),

                    # Date range
                    html.Label("Date Range:", className="fw-bold"),
                    dcc.DatePickerRange(
                        id="date-range",
                        start_date="2022-01-01",
                        end_date="2023-12-31",
                        className="mb-3",
                    ),

                    html.Hr(),

                    # Forecast settings
                    html.Label("Forecast Horizon (days):", className="fw-bold"),
                    dcc.Slider(
                        id="forecast-horizon",
                        min=7,
                        max=90,
                        step=7,
                        value=30,
                        marks={7: "7", 30: "30", 60: "60", 90: "90"},
                        className="mb-3",
                    ),

                    html.Label("Forecast Model:", className="fw-bold"),
                    dcc.Dropdown(
                        id="forecast-model",
                        options=[
                            {"label": "ARIMA", "value": "arima"},
                            {"label": "ETS", "value": "ets"},
                            {"label": "Machine Learning", "value": "ml"},
                            {"label": "Ensemble", "value": "ensemble"},
                        ],
                        value="ensemble",
                        clearable=False,
                        className="mb-3",
                    ),

                    html.Hr(),

                    # Scenario controls
                    html.Label("Demand Change (%):", className="fw-bold"),
                    dcc.Slider(
                        id="demand-change",
                        min=-30,
                        max=30,
                        step=5,
                        value=0,
                        marks={-30: "-30%", 0: "0%", 30: "+30%"},
                        className="mb-3",
                    ),

                    html.Label("Capacity Change (%):", className="fw-bold"),
                    dcc.Slider(
                        id="capacity-change",
                        min=-30,
                        max=30,
                        step=5,
                        value=0,
                        marks={-30: "-30%", 0: "0%", 30: "+30%"},
                        className="mb-3",
                    ),

                    html.Hr(),

                    # Action buttons
                    dbc.Button(
                        "Run Analysis",
                        id="run-analysis-btn",
                        color="primary",
                        className="w-100 mb-2",
                    ),
                    dbc.Button(
                        "Generate Report",
                        id="generate-report-btn",
                        color="secondary",
                        outline=True,
                        className="w-100",
                    ),
                ]
            ),
        ],
        className="mb-4",
    )


def create_kpi_cards() -> dbc.Row:
    """Create the KPI summary cards."""
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Weekly Demand", className="card-subtitle text-muted"),
                            html.H3(id="kpi-demand", children="--", className="card-title"),
                            html.P("referrals/week", className="card-text text-muted small"),
                        ]
                    ),
                    className="text-center",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Weekly Capacity", className="card-subtitle text-muted"),
                            html.H3(id="kpi-capacity", children="--", className="card-title"),
                            html.P("appointments/week", className="card-text text-muted small"),
                        ]
                    ),
                    className="text-center",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Utilisation", className="card-subtitle text-muted"),
                            html.H3(id="kpi-utilisation", children="--", className="card-title"),
                            html.P("capacity used", className="card-text text-muted small"),
                        ]
                    ),
                    className="text-center",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Avg Wait Time", className="card-subtitle text-muted"),
                            html.H3(id="kpi-wait-time", children="--", className="card-title"),
                            html.P("days", className="card-text text-muted small"),
                        ]
                    ),
                    className="text-center",
                ),
                md=3,
            ),
        ],
        className="mb-4",
    )


def create_main_content() -> dbc.Col:
    """Create the main content area with charts."""
    return dbc.Col(
        [
            # KPI Cards
            create_kpi_cards(),

            # Charts in tabs
            dbc.Card(
                [
                    dbc.CardHeader(
                        dbc.Tabs(
                            [
                                dbc.Tab(label="Demand Forecast", tab_id="tab-forecast"),
                                dbc.Tab(label="Capacity Analysis", tab_id="tab-capacity"),
                                dbc.Tab(label="Queue Simulation", tab_id="tab-simulation"),
                                dbc.Tab(label="Scenario Comparison", tab_id="tab-scenarios"),
                            ],
                            id="chart-tabs",
                            active_tab="tab-forecast",
                        )
                    ),
                    dbc.CardBody(
                        [
                            dcc.Loading(
                                id="loading-chart",
                                type="default",
                                children=[
                                    html.Div(id="chart-content"),
                                ],
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),

            # Additional insights
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Model Performance"),
                                dbc.CardBody(id="model-metrics"),
                            ]
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Recommendations"),
                                dbc.CardBody(id="recommendations"),
                            ]
                        ),
                        md=6,
                    ),
                ]
            ),
        ],
        md=9,
    )


def create_layout() -> html.Div:
    """Create the main application layout."""
    return html.Div(
        [
            # Navigation
            create_navbar(),

            # Main content
            dbc.Container(
                [
                    html.Br(),
                    dbc.Row(
                        [
                            # Sidebar
                            dbc.Col(create_sidebar(), md=3),

                            # Main content
                            create_main_content(),
                        ]
                    ),
                ],
                fluid=True,
            ),

            # Store components for data
            dcc.Store(id="demand-data-store"),
            dcc.Store(id="capacity-data-store"),
            dcc.Store(id="forecast-data-store"),
            dcc.Store(id="simulation-data-store"),

            # Location for routing
            dcc.Location(id="url", refresh=False),
        ]
    )
