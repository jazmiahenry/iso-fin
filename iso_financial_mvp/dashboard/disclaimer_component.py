import dash
from dash import html
import dash_bootstrap_components as dbc

from iso_financial_mvp.report_generator.disclaimer_template import get_standard_disclaimer

def create_disclaimer_component(id="disclaimer-collapse", is_open=False):
    """
    Create a collapsible disclaimer component for the dashboard
    
    Args:
        id: Component ID for the collapse element
        is_open: Whether the disclaimer is initially expanded
        
    Returns:
        Dash component with the disclaimer
    """
    disclaimer_text = get_standard_disclaimer()
    
    # Replace newlines with HTML line breaks for better formatting
    formatted_text = disclaimer_text.replace('\n\n', '<br><br>')
    
    return html.Div([
        dbc.Button(
            "View Disclaimer",
            id=f"{id}-button",
            color="secondary",
            size="sm",
            className="mb-3"
        ),
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody([
                    html.H5("IMPORTANT DISCLAIMER: NOT FINANCIAL ADVICE", className="card-title"),
                    html.Div([
                        html.P(paragraph, style={"margin-bottom": "1rem"})
                        for paragraph in disclaimer_text.split('\n\n') if paragraph.strip()
                    ])
                ]),
                className="disclaimer-card"
            ),
            id=id,
            is_open=is_open
        )
    ], className="disclaimer-container")

def init_disclaimer_callback(app):
    """
    Initialize callback for the disclaimer collapse toggle
    
    Args:
        app: Dash app instance
    """
    @app.callback(
        dash.dependencies.Output("disclaimer-collapse", "is_open"),
        [dash.dependencies.Input("disclaimer-collapse-button", "n_clicks")],
        [dash.dependencies.State("disclaimer-collapse", "is_open")],
    )
    def toggle_disclaimer(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open