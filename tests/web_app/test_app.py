"""
Tests for the web application module.
"""


from demand_capacity_platform.web_app import create_app


class TestDashApp:
    """Tests for Dash application creation."""

    def test_create_app(self):
        """Test that the app can be created."""
        app = create_app(debug=False)

        assert app is not None
        assert app.title == "Demand & Capacity Platform"

    def test_create_app_with_custom_title(self):
        """Test app creation with custom title."""
        app = create_app(debug=False, title="Custom Title")

        assert app.title == "Custom Title"

    def test_app_layout_exists(self):
        """Test that app has a layout."""
        app = create_app(debug=False)

        assert app.layout is not None
