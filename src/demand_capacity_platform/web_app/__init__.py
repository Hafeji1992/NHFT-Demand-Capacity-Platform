"""
Web Application Module.

Provides a Dash-based web interface for visualizing forecasts,
scenarios, and utilisation metrics.
"""

from demand_capacity_platform.web_app.app import create_app

__all__ = ["create_app"]
