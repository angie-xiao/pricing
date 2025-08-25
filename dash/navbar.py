# navbar.py
import dash_bootstrap_components as dbc
from dash import dcc


def get_navbar():
    nav = dbc.Nav(
        [
            dbc.NavItem(
                dcc.Link("Home", href="/", className="nav-link", id="nav-home")
            ),
            dbc.NavItem(
                dcc.Link(
                    "overview",
                    href="/overview",
                    className="nav-link",
                    id="nav-overview",
                )
            ),
            dbc.NavItem(
                dcc.Link(
                    "Predictive",
                    href="/predictive",
                    className="nav-link",
                    id="nav-predictive",
                )
            ),
            dbc.NavItem(
                dcc.Link("FAQ", href="/faq", className="nav-link", id="nav-faq")
            ),
        ],
        className="ms-auto",
        pills=True,
    )

    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("Optimal Pricing", href="/", className="fw-bold"),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(nav, id="navbar-collapse", navbar=True),
            ],
            fluid=True,
        ),
        color="light",
        light=True,
        className="shadow-sm",
    )
