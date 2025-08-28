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
                    "Overview",
                    href="/overview",
                    className="nav-link",
                    id="nav-overview",
                )
            ),             
            # dbc.NavItem(
            #     dcc.Link(
            #         "Descriptive",
            #         href="/descriptive",
            #         className="nav-link",
            #         id="nav-descriptive",
            #     )
            # ), 
            dbc.NavItem(
                dcc.Link(
                    "Opportunities",
                    href="/opps",
                    className="nav-link",
                    id="nav-opps",
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
                dbc.NavbarBrand("Optima", href="/", className="fw-bold"),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(nav, id="navbar-collapse", navbar=True),
            ],
            fluid=True,
        ),
        color="light",
        # light=True,   # mac
        dark=False,     # windows
        className="shadow-sm",
    )
