"""
Dash Web Application Module.

Main entry point for the Demand and Capacity Platform web interface.
"""

import logging

import dash_bootstrap_components as dbc
from dash import Dash

from demand_capacity_platform.web_app.callbacks import register_callbacks
from demand_capacity_platform.web_app.layouts import create_layout

logger = logging.getLogger(__name__)


def create_app(
    debug: bool = False,
    title: str = "Demand & Capacity Platform"
) -> Dash:
    """
    Create and configure the Dash application.

    Args:
        debug: Whether to run in debug mode.
        title: Application title.

    Returns:
        Configured Dash application instance.

    Example:
        >>> app = create_app(debug=True)
        >>> app.run(port=8050)
    """
    logger.info("Creating Dash application")

    # Initialize Dash app with Bootstrap theme
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title=title,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ]
    )

    # Set layout
    app.layout = create_layout()

    # Register callbacks
    register_callbacks(app)

    logger.info("Dash application created successfully")

    return app


def main():
    """Main entry point for running the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    app = create_app(debug=True)
    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
