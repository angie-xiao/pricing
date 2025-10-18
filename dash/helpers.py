# helpers.py
# ---------------------------------------------------------------------
# Class-first helpers for the pricing dashboard (with BC wrappers)
# ---------------------------------------------------------------------

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dash import html
import dash_bootstrap_components as dbc
from dash.dash_table import FormatTemplate

# =====================================================================
#                        Constants / Styling
# =====================================================================


class Style:
    OVERVIEW_ACCENT = {"color": "#DAA520"}
    HOME_ACCENT = {
        "padding": "24px 0",
        "color": "#DAA520",
        "marginLeft": "50px",
        "marginRight": "50px",
    }
    HOME_MUTED = {"color": "#5f6b7a", "marginLeft": "50px", "marginRight": "50px"}
    HOME_SECTION_STYLE = {
        "padding": "14px 0",
        "marginLeft": "50px",
        "marginRight": "50px",
    }
    HOME_CARD_STYLE = {
        "border": "1px solid #e9eef5",
        "borderRadius": "14px",
        "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
    }

    # Opps page spacing
    OPP_LEFT_INSET = "50px"
    OPP_GAP_H1_TO_TABLE = "32px"
    OPP_GAP_TABLE_TO_H3 = "36px"
    OPP_GAP_H3_TO_GRAPH = "8px"


class Paths:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))  # .../pricing/dash
    PROJECT_BASE = os.path.dirname(BASE_DIR)  # .../pricing


# =====================================================================
#                         Caching / Build
# =====================================================================


class Cache:
    @staticmethod
    def files_sig(paths: List[str], top_n: int = 10, version: str = "v1") -> str:
        parts: List[str] = []
        for p in paths:
            ap = os.path.abspath(p)
            try:
                st = os.stat(ap)
                parts.append(f"{ap}:{st.st_mtime_ns}:{st.st_size}")
            except FileNotFoundError:
                parts.append(f"{ap}:NA")
        sig_str = "|".join(parts) + f":top{top_n}:ver:{version}"
        return hashlib.sha1(sig_str.encode()).hexdigest()

    @staticmethod
    def code_sig() -> str:
        """
        Hash of: our version tag + source of _make_weights + file mtimes.
        Any change to weighting logic or this file will bust the cache.
        """
        import hashlib, inspect, os, built_in_logic

        h = hashlib.sha256()
        # 1) explicit version tag
        ver = getattr(built_in_logic, "WEIGHT_LOGIC_VERSION", "unknown").encode()
        h.update(ver)

        # 2) function source (most direct way to detect logic changes)
        try:
            src = inspect.getsource(built_in_logic._make_weights).encode()
            h.update(src)
        except Exception:
            h.update(b"no-source")

        # 3) file mtime as a fallback
        try:
            fp = inspect.getfile(built_in_logic)
            mtime = str(os.path.getmtime(fp)).encode()
            h.update(mtime)
        except Exception:
            pass

        return h.hexdigest()[:12]


    @staticmethod
    def build_frames_with_cache(
        base_dir: str,
        data_folder: str = "data",
        pricing_file: str = "pricing.csv",
        product_file: str = "products.csv",
        top_n: int = 10,
        force_rebuild: bool = True,
        param_search_kwargs: dict = None  # Add this parameter
    ) -> Dict[str, pd.DataFrame]:
        cache_dir = os.path.join(base_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)

        pricing_path = os.path.join(base_dir, data_folder, pricing_file)
        product_path = os.path.join(base_dir, data_folder, product_file)
        
        # Include parameters in signature
        param_sig = ""
        if param_search_kwargs:
            param_items = sorted(param_search_kwargs.items())
            param_sig = "|".join(f"{k}:{v}" for k, v in param_items)
        
        sig = Cache.files_sig(
            [pricing_path, product_path], 
            top_n=top_n, 
            version=f"{Cache.code_sig()}|{param_sig}"
        )

        cache_fp = os.path.join(cache_dir, f"frames_{sig}.pkl")

        # Clear cache if force_rebuild
        if force_rebuild and os.path.exists(cache_fp):
            os.remove(cache_fp)

        if force_rebuild or not os.path.exists(cache_fp):
            from built_in_logic import PricingPipeline
  

            frames = PricingPipeline.from_csv_folder(
                base_dir,
                data_folder="data",
                pricing_file="pricing.csv",
                product_file="products.csv",
                top_n=10,
            )
            # Only save to cache if not forcing rebuild
            if not force_rebuild:
                with open(cache_fp, "wb") as f:
                    pickle.dump(frames, f)
            
            return frames

        # Load from cache
        with open(cache_fp, "rb") as f:
            return pickle.load(f)


# =====================================================================
#                         Data engineering 
# =====================================================================
class DataEng:
    @staticmethod
    def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = out.columns.str.strip().str.lower()
        return out
 
    @staticmethod
    def compute_product_series(df, tag_col="tag", var_col="variation"):
        if tag_col in df.columns and var_col in df.columns:
            return (df[tag_col].astype(str) + " " + df[var_col].astype(str)).str.upper()
        elif tag_col in df.columns:
            return df[tag_col].astype(str).str.upper()
        elif var_col in df.columns:
            return df[var_col].astype(str).str.upper()
        else:
            # fall back to asin if no descriptive cols
            return df["asin"].astype(str)

    @staticmethod
    def nearest_row_at_price(
        prod_df: pd.DataFrame, price: float, price_col: str = "asp"
    ) -> Optional[pd.Series]:
        if (
            prod_df is None
            or prod_df.empty
            or pd.isna(price)
            or price_col not in prod_df
        ):
            return None
        try:
            idx = (prod_df[price_col] - float(price)).abs().idxmin()
            return prod_df.loc[idx]
        except Exception:
            return None

    @staticmethod
    def pick_best_by_group(
        df: pd.DataFrame, group_col: str, value_col: str
    ) -> pd.DataFrame:
        idx = df.groupby(group_col)[value_col].idxmax()
        return df.loc[idx].reset_index(drop=True)

    def make_products_lookup(*dfs: pd.DataFrame) -> pd.DataFrame:
        """
        Build a robust product ↔ key lookup from any frames that already contain both.
        """
        pieces: List[pd.DataFrame] = []
        for df in dfs:
            if df is None or len(df) == 0:
                continue
            cols = set(df.columns)
            if {"product", "asin"}.issubset(cols):
                pieces.append(df[["product", "asin"]].copy())

        if not pieces:
            raise KeyError("No source frame had both ['product','asin'].")

        lookup = (
            pd.concat(pieces, ignore_index=True)
            .dropna(subset=["product", "asin"])
            .astype({"asin": str})
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return lookup


# =====================================================================
#                        Opportunities Page
# =====================================================================


class OppsTable:
    @staticmethod
    def ensure_days_valid_column(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        if "days_asp_valid" not in out.columns:
            out["days_asp_valid"] = (
                out["days_sold"] if "days_sold" in out.columns else 1
            )
        return out

    @staticmethod
    def build_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []

        cols_in_order = [
            "asin",
            "product",
            "asp",
            "days_asp_valid",
            "days_sold",
            "shipped_units",
            "daily_units",
            "revenue_actual",
            "daily_rev",
            "revenue_pred_0.025",
            "revenue_pred_0.5",
            "revenue_pred_0.975",
            "pred_0.025",
            "pred_0.5",
            "pred_0.975",
        ]
        present = [c for c in cols_in_order if c in df.columns]
        df = df[present].copy()

        integer_columns = {
            "days_asp_valid",
            "days_sold",
            "shipped_units",
            "daily_units",
        }
        money_columns = {
            "asp",
            "revenue_actual",
            "daily_rev",
            "revenue_pred_0.025",
            "revenue_pred_0.5",
            "revenue_pred_0.975",
            "pred_0.025",
            "pred_0.5",
            "pred_0.975",
        }

        cols: List[Dict[str, Any]] = []
        for c in df.columns:
            if c in integer_columns:
                cols.append(
                    {
                        "name": c.replace("_", " ").title(),
                        "id": c,
                        "type": "numeric",
                        "format": {"specifier": ",d"},
                    }
                )
            elif c in money_columns:
                cols.append(
                    {
                        "name": c.replace("_", " ").title(),
                        "id": c,
                        "type": "numeric",
                        "format": FormatTemplate.money(2),
                    }
                )
            else:
                cols.append({"name": c.replace("_", " ").title(), "id": c})
        return cols


# =====================================================================
#                   Home page small UI pieces
# =====================================================================


class HomeUI:
    @staticmethod
    def step_card(title: str, desc: str) -> dbc.Col:
        P_STYLE = {
            "color": "#5f6b7a",
            "marginLeft": "10px",
            "marginRight": "10px",
            "marginTop": "10px",
        }
        return dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [html.H5(title), html.P(desc, style=P_STYLE)],
                    className="d-flex flex-column",
                ),
                style=Style.HOME_CARD_STYLE,
                className="h-100",
            ),
        )

    @staticmethod
    def info_card(title: str, text: str) -> dbc.Col:
        P_STYLE = {
            "color": "#5f6b7a",
            "marginLeft": "10px",
            "marginRight": "10px",
            "marginTop": "10px",
        }
        return dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [html.H5(title), html.P(text, style=P_STYLE)],
                    className="d-flex flex-column",
                ),
                style=Style.HOME_CARD_STYLE,
                className="h-100",
            ),
        )

    @staticmethod
    def footer_two_lines(
        signature_handle: str = "@aqxiao",
        link_text: str = "github.com/angie-xiao",
        link_href: Optional[str] = "https://github.com/angie-xiao",
    ) -> html.Div:
        return html.Div(
            [
                html.Div(
                    [html.Span("made with ♥️ | "), html.Span(html.I(signature_handle))],
                    style={"marginBottom": "4px"},
                ),
                html.A(
                    link_text,
                    href=link_href,
                    style={"textDecoration": "none", "color": "#ac274f"},
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "height": "80px",
                "fontSize": "0.8em",
                "color": "#ac274f",
                "backgroundColor": "#f3f3f3",
                "borderRadius": "0",
                "width": "100%",
                "marginTop": "40px",
            },
        )


# =====================================================================
#                   Overview page UI & metrics
# =====================================================================


class OverviewUI:
    @staticmethod
    def kpi_card(
        id_title: str,
        title: str,
        id_value: str,
        bg: str = "#f3f0f0",
        id_subtext: Optional[str] = None,
    ) -> dbc.Col:
        return dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            id=id_title,
                            className="kpi-title",
                            style={
                                "color": "#121212",
                                "textAlign": "center",
                                "marginBottom": "10px",
                                "marginTop": "10px",
                                "whiteSpace": "nowrap",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                            },
                        ),
                        html.Div(
                            style={
                                "height": "4px",
                                "width": "44px",
                                "margin": "2px auto 8px",
                                "borderRadius": "999px",
                                "background": "linear-gradient(90deg,#DAA520,#F0C64C)",
                                "opacity": 0.9,
                            }
                        ),
                        html.H2(
                            title,
                            className="kpi-eyebrow",
                            style={
                                "color": "#121212",
                                "textAlign": "center",
                                "fontSize": "18px",
                                "letterSpacing": ".14em",
                                "fontWeight": 700,
                            },
                        ),
                        html.H1(
                            id=id_value,
                            className="kpi-value",
                            style={
                                "color": "#DAA520",
                                "textAlign": "center",
                                "fontSize": "44px",
                                "fontWeight": 800,
                            },
                        ),
                        html.Div(
                            id=id_subtext if id_subtext else f"{id_value}-subtext",
                            className="kpi-subtext text-muted",
                            style={
                                "textAlign": "center",
                                "fontSize": "15px",
                                "marginTop": "6px",
                                "lineHeight": "1.25",
                                "minHeight": "34px",
                            },
                        ),
                    ],
                    className="d-flex flex-column justify-content-start",
                    style={"gap": "6px", "padding": "22px 18px"},
                ),
                className="h-100 shadow-sm border rounded-4",
                style={
                    "backgroundColor": bg,
                    "padding": "12px 0",
                    "borderColor": "rgba(17,24,39,0.08)",
                    "boxShadow": "0 1px 2px rgba(16,24,40,.03), 0 4px 8px rgba(16,24,40,.04)",
                    "backgroundImage": "radial-gradient(160% 80% at 50% 0%, rgba(218,165,32,.03), transparent 45%)",
                },
            ),
            width=3,
            className="kpi-card",
        )


class Formatters:
    @staticmethod
    def money(x: Any) -> str:
        try:
            return f"${float(x):,.0f}"
        except Exception:
            return "—"

    @staticmethod
    def units(x: Any) -> str:
        try:
            return f"{float(x):,.0f}"
        except Exception:
            return "—"

    @staticmethod
    def signed_units(x: Any) -> str:
        try:
            xf = float(x)
            sign = "+" if np.isfinite(xf) and xf >= 0 else "-"
            return "—" if not np.isfinite(xf) else f"{sign}{abs(xf):,.0f}"
        except Exception:
            return "—"

    @staticmethod
    def signed_money(x: Any) -> str:
        try:
            xf = float(x)
            sign = "+" if np.isfinite(xf) and xf >= 0 else "-"
            return "—" if not np.isfinite(xf) else f"{sign}${abs(xf):,.0f}"
        except Exception:
            return "—"

    @staticmethod
    def date(dt: Any) -> str:
        try:
            dt = pd.to_datetime(dt)
            return "—" if pd.isna(dt) else dt.strftime("%Y-%m-%d")
        except Exception:
            return "—"


class Metrics:
 
    @staticmethod
    def model_fit_units(prod_df: pd.DataFrame) -> Tuple[str, str]:
        """(value_text, subtext) RMSE on daily revenue (P50) + weighted relative error."""
        if prod_df is None or prod_df.empty:
            return "—", ""

        df = prod_df.copy()

        # Accept either 'daily_rev' or 'revenue'
        if "daily_rev" not in df.columns:
            if "revenue" in df.columns:
                df["daily_rev"] = pd.to_numeric(df["revenue"], errors="coerce")
            else:
                return "—", ""

        # Need prediction
        if "revenue_pred_0.5" not in df.columns:
            return "—", ""

        if "asp" not in df.columns and "price" in df.columns:
            df["asp"] = pd.to_numeric(df["price"], errors="coerce")

        df = df[["asp", "daily_rev", "revenue_pred_0.5"]].dropna()
        if df.empty:
            return "—", ""

        # Aggregate at price to stabilize duplicates
        df = df.groupby("asp", as_index=False).agg(
            daily_rev=("daily_rev", "mean"),
            pred_rev=("revenue_pred_0.5", "mean"),
        )
        if df.empty:
            return "—", ""

        # RMSE
        diff2 = (df["pred_rev"] - df["daily_rev"]) ** 2
        rmse = float(np.sqrt(np.nanmean(diff2))) if len(diff2) else np.nan

        # Weighted relative error (closer to MAPE but avoids div-by-zero)
        mask = df["daily_rev"] > 0
        w = df["daily_rev"].where(mask, np.nan)
        pct = (df["pred_rev"] - df["daily_rev"]) / df["daily_rev"]
        rel = float(np.nansum(pct * w) / np.nansum(w)) if np.nansum(w) > 0 else np.nan

        if np.isnan(rmse):
            return "—", ""

        sub = f"{rel:+.0%} vs actual" if np.isfinite(rel) else ""
        return f"${rmse:,.0f}", sub
    
    
    @staticmethod
    def update_elasticity_kpi_by_product(
        product_name: str, elast_df: pd.DataFrame
    ) -> Tuple[str, str]:
        try:
            row = elast_df.loc[elast_df["product"] == product_name]
            if row.empty or "ratio" not in row:
                return "—", ""

            ratio = float(row["ratio"].iloc[0])

            # Use elasticity_score instead of pct
            elasticity_score = float(row["elasticity_score"].iloc[0])
            value_text = f"{ratio:,.2f}"

            # Calculate elasticity tier
            elasticity_tier = int(round(elasticity_score))
            top_share = max(1, elasticity_tier)

            subtext = (
                "Top ~{0}% most ELASTIC".format(top_share)
                if elasticity_score >= 50
                else "Top ~{0}% most INELASTIC".format(top_share)
            )
            return value_text, subtext

        except Exception:
            return "—", ""


class Scenario:
    @staticmethod
    def table(prod_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scenario table for Revenue & Shipped Units.
        For q in {0.025, 0.5, 0.975}, find ASP that maximizes revenue_pred_q and report
        Price / Revenue / Units (at that ASP).
        """
        cols = ["case", "price", "revenue", "units"]
        if prod_df is None or prod_df.empty or "asp" not in prod_df.columns:
            return pd.DataFrame(
                [{"case": "—", "price": "—", "revenue": "—", "units": "—"}],
                columns=cols,
            )

        rows: List[Dict[str, Any]] = []
        spec = [
            ("revenue_pred_0.025", "units_pred_0.025", "Conservative"),
            ("revenue_pred_0.5", "units_pred_0.5", "Expected"),
            ("revenue_pred_0.975", "units_pred_0.975", "Optimistic"),
        ]
        for rev_col, units_col, label in spec:
            if rev_col not in prod_df.columns:
                continue
            sub = prod_df.dropna(subset=["asp", rev_col]).copy()
            if sub.empty:
                continue
            try:
                idx = sub[rev_col].idxmax()
                row = sub.loc[idx]
            except Exception:
                continue

            price = row.get("asp", np.nan)
            rev = row.get(rev_col, np.nan)
            units = row.get(units_col, np.nan) if units_col in sub.columns else np.nan
            rows.append(
                {
                    "case": label,
                    "price": (
                        f"${float(price):,.2f}" if np.isfinite(float(price)) else "—"
                    ),
                    "revenue": (
                        f"${float(rev):,.0f}" if np.isfinite(float(rev)) else "—"
                    ),
                    "units": (
                        f"{float(units):,.0f}" if np.isfinite(float(units)) else "—"
                    ),
                }
            )
        if not rows:
            return pd.DataFrame(
                [{"case": "—", "price": "—", "revenue": "—", "units": "—"}],
                columns=cols,
            )
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    def annualized_kpis_signed(
        asin: str,
        best50_df: pd.DataFrame,
        curr_price_df: pd.DataFrame,
        all_gam_results: pd.DataFrame,
    ) -> Tuple[str, str]:
        """
        Return (annualized_units_delta, annualized_revenue_delta) as signed strings.
        Opportunity = (recommended - current) * 365

        - Looks up recommended daily rev/units from best50_df (or curve fallback)
        - Looks up current daily rev/units by nearest ASP on curve
        - Computes signed annual deltas
        """
        try:
            if not asin or not isinstance(asin, str):
                return "—", "—"
            if best50_df is None or curr_price_df is None or all_gam_results is None:
                return "—", "—"

            # --- Recommended row ---
            best_row = best50_df.loc[best50_df["asin"] == asin]
            if best_row.empty:
                return "—", "—"

            if {"asp", "revenue_pred_0.5", "units_pred_0.5"}.issubset(best_row.columns):
                daily_rev_best = float(best_row["revenue_pred_0.5"].iloc[0])
                daily_units_best = float(best_row["units_pred_0.5"].iloc[0])
                rec_price = float(best_row["asp"].iloc[0])
            elif {"price", "revenue_pred_0.5", "units_pred_0.5"}.issubset(best_row.columns):
                daily_rev_best = float(best_row["revenue_pred_0.5"].iloc[0])
                daily_units_best = float(best_row["units_pred_0.5"].iloc[0])
                rec_price = float(best_row["price"].iloc[0])
            else:
                return "—", "—"

            # --- Current price ---
            if "asin" in curr_price_df.columns and "current_price" in curr_price_df.columns:
                cp_series = curr_price_df.loc[curr_price_df["asin"] == asin, "current_price"]
                curr_price = float(cp_series.iloc[0]) if len(cp_series) else None
            else:
                curr_price = None
            if curr_price is None or not np.isfinite(curr_price):
                return "—", "—"

            # --- Curve for this ASIN ---
            prod_curve = (
                all_gam_results.loc[all_gam_results["asin"] == asin]
                if "asin" in all_gam_results.columns
                else all_gam_results
            )
            if prod_curve.empty:
                return "—", "—"

            need_cols = {"asp", "revenue_pred_0.5", "units_pred_0.5"}
            if not need_cols.issubset(prod_curve.columns):
                return "—", "—"

            # Recommended metrics from curve (for consistency)
            idx_best = (prod_curve["asp"] - rec_price).abs().idxmin()
            daily_rev_best = float(prod_curve.loc[idx_best, "revenue_pred_0.5"])
            daily_units_best = float(prod_curve.loc[idx_best, "units_pred_0.5"])

            # Current metrics from curve
            idx_curr = (prod_curve["asp"] - curr_price).abs().idxmin()
            daily_rev_curr = float(prod_curve.loc[idx_curr, "revenue_pred_0.5"])
            daily_units_curr = float(prod_curve.loc[idx_curr, "units_pred_0.5"])

            # --- Signed annual deltas ---
            units_delta_annual = (daily_units_best - daily_units_curr) * 365.0
            rev_delta_annual = (daily_rev_best - daily_rev_curr) * 365.0

            return (
                Formatters.signed_units(units_delta_annual),
                Formatters.signed_money(rev_delta_annual),
            )

        except Exception as e:
            print(f"Error in annualized_kpis_signed: {e}")
            return "—", "—"


class Badges:
    @staticmethod
    def robustness(prod_df: pd.DataFrame):
        if prod_df is None or prod_df.empty:
            return ""

        def _peak_asp(col: str) -> float:
            if col not in prod_df or prod_df[col].isna().all():
                return np.nan
            idx = prod_df[col].idxmax()
            try:
                return float(prod_df.loc[idx, "asp"])
            except Exception:
                return np.nan

        p_low, p_mid, p_high = (
            _peak_asp("revenue_pred_0.025"),
            _peak_asp("revenue_pred_0.5"),
            _peak_asp("revenue_pred_0.975"),
        )
        if np.isnan([p_low, p_mid, p_high]).any() or (not p_mid):
            spread_score = 0.0
        else:
            align_spread = max(p_low, p_mid, p_high) - min(p_low, p_mid, p_high)
            spread_score = float(np.exp(-align_spread / (0.1 * p_mid)))

        elasticity_score = 0.0
        if "elasticity" in prod_df.columns and p_mid and not np.isnan(p_mid):
            try:
                row = prod_df.loc[
                    (prod_df["asp"] == p_mid) & prod_df["elasticity"].notna()
                ]
                if not row.empty:
                    el_mid = float(row["elasticity"].iloc[0])
                    el_min = float(prod_df["elasticity"].min())
                    el_max = float(prod_df["elasticity"].max())
                    elasticity_score = (
                        (1.0 - (el_mid - el_min) / (el_max - el_min))
                        if el_max > el_min
                        else 1.0
                    )
            except Exception:
                pass

        try:
            n_distinct_prices = int(prod_df["asp"].nunique(dropna=True))
        except Exception:
            n_distinct_prices = prod_df.shape[0]

        data_strength = 1.0 - float(np.exp(-n_distinct_prices / 6.0))
        credibility_multiplier = 0.6 + 0.4 * data_strength
        base_score = 0.4 * spread_score + 0.6 * elasticity_score
        final_score = base_score * credibility_multiplier

        label, color = ("Weak", "danger")
        if final_score >= 0.70:
            label, color = ("Strong", "success")
        elif final_score >= 0.45:
            label, color = ("Medium", "warning")

        return dbc.Badge(
            f"Confidence: {label}", color=color, pill=True, className="px-3 py-2"
        )


class Notes:
    @staticmethod
    def coverage(prod_df: pd.DataFrame):
        if prod_df is None or prod_df.empty:
            return ""
        n_points = int(len(prod_df))
        if {"revenue_actual", "revenue_pred_0.025", "revenue_pred_0.975"}.issubset(
            prod_df.columns
        ):
            within = (
                (prod_df["revenue_actual"] >= prod_df["revenue_pred_0.025"])
                & (prod_df["revenue_actual"] <= prod_df["revenue_pred_0.975"])
            ).mean()
        else:
            within = np.nan
        return html.Div(
            [
                html.Div(
                    [
                        "Based on ",
                        html.Span(
                            f"{n_points}",
                            style={
                                "color": Style.OVERVIEW_ACCENT["color"],
                                "fontWeight": 600,
                            },
                        ),
                        " historical points; ",
                        html.Span(
                            f"{within*100:,.0f}%" if np.isfinite(within) else "—",
                            style={
                                "color": Style.OVERVIEW_ACCENT["color"],
                                "fontWeight": 600,
                            },
                        ),
                        " of actual revenue outcomes fall within the shown range.",
                    ],
                    style={"textAlign": "center"},
                ),
            ],
            style={"marginTop": "78px"},
        )


# =====================================================================
#                        Overview Page Helpers
# =====================================================================


class OverviewHelpers:
    """Small, testable helpers used by the Overview callback."""

    @staticmethod
    def empty_overview_payload(viz):
        empty_fig = viz.empty_fig("Select a product")
        return (
            "",
            "—",
            "",
            "",
            "—",
            "",
            "—",
            "",
            "—",
            "",
            "",
            "—",
            "",
            "—",
            "",
            "—",
            "",
            empty_fig,
            [{"case": "—", "price": "—", "revenue": "—", "units": "—"}],
            "",
        )

    @staticmethod
    def date_kpis(meta: Dict[str, Any]) -> Tuple[str, str]:
        start_date = pd.to_datetime(meta.get("data_start"))
        end_date = pd.to_datetime(meta.get("data_end"))
        if pd.notna(start_date) and pd.notna(end_date):
            num_days = (end_date - start_date).days + 1
            return f"{num_days:,}", f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"
        return "—", "—"

    @staticmethod
    def display_name(asin: str, products_lookup: pd.DataFrame) -> str:
        try:
            return products_lookup.loc[products_lookup["asin"] == asin, "product"].iloc[
                0
            ]
        except Exception:
            return ""

    @staticmethod
    def current_price_text(asin: str, curr_price_df: pd.DataFrame) -> str:
        cp = curr_price_df.loc[curr_price_df["asin"] == asin, "current_price"]
        return f"${float(cp.iloc[0]):,.2f}" if len(cp) else "—"

    @staticmethod 
    def recommended_price_text(asin: str, best50_df: pd.DataFrame) -> str:
        row = best50_df.loc[best50_df["asin"] == asin]
        if not len(row):
            return "—"
        if "asp" in row.columns:
            val = row["asp"].iloc[0]
        elif "price" in row.columns:
            val = row["price"].iloc[0]
        else:
            return "—"
        try:
            return f"${float(val):,.2f}"
        except Exception:
            return "—"


    @staticmethod
    def elasticity_texts(
        product_name: str, elasticity_df: pd.DataFrame
    ) -> Tuple[str, str]:
        return Metrics.update_elasticity_kpi_by_product(product_name, elasticity_df)

    @staticmethod
    def filter_product_rows(asin: str, all_gam_results: pd.DataFrame) -> pd.DataFrame:
        return all_gam_results[all_gam_results["asin"] == asin]

    @staticmethod
    def pred_graph(viz, filt_df: pd.DataFrame):
        return (
            viz.gam_results(filt_df)
            # viz.predictive_from_frames(filt_df, aggregate="mean")
        )

    @staticmethod
    def scenario_records(filt, include_weighted=True):
        """Enhanced scenario table with weighted predictions"""
        if filt is None or filt.empty:
            return []

        scenarios = []

        # Add recommended scenarios
        scenarios.extend(
            [
                {
                    "case": "Conservative",
                    "price": f"${filt.loc[filt['revenue_pred_0.025'].idxmax(), 'price']:.2f}",
                    "revenue": f"${filt['revenue_pred_0.025'].max():,.0f}",
                },
                {
                    "case": "Expected",
                    "price": f"${filt.loc[filt['revenue_pred_0.5'].idxmax(), 'price']:.2f}",
                    "revenue": f"${filt['revenue_pred_0.5'].max():,.0f}",
                },
                {
                    "case": "Optimistic",
                    "price": f"${filt.loc[filt['revenue_pred_0.975'].idxmax(), 'price']:.2f}",
                    "revenue": f"${filt['revenue_pred_0.975'].max():,.0f}",
                },
            ]
        )

        # Add weighted prediction if available
        if include_weighted and "weighted_pred" in filt.columns:
            weighted_best = filt.loc[filt["weighted_pred"].idxmax()]
            scenarios.insert(
                1,
                {
                    "case": "Recommended (Weighted)",
                    "price": f"${weighted_best['price']:.2f}",
                    "revenue": f"${weighted_best['revenue_pred_0.5']:,.0f}",
                },
            )

        # Insert current price scenario at the beginning if available
        if "current_price" in filt.columns:
            current_row = filt.iloc[0]  # Get first row for current scenario
            scenarios.insert(
                0,
                {
                    "case": "Current Price",
                    "price": f"${current_row['current_price']:.2f}",
                    "revenue": f"${current_row['revenue_pred_0.5']:,.0f}",
                },
            )

        return scenarios

    @staticmethod
    def annualized_kpis(
        asin: str,
        best50_df: pd.DataFrame,
        curr_price_df: pd.DataFrame,
        all_gam_results: pd.DataFrame,
    ) -> Tuple[str, str]:
        return Scenario.annualized_kpis_signed(
            asin, best50_df, curr_price_df, all_gam_results
        )


# =====================================================================
# Backwards-compatibility wrappers (keep existing imports working)
# =====================================================================


# cache wrappers
def build_frames_with_cache(*args, **kwargs):
    return Cache.build_frames_with_cache(*args, **kwargs)


# data eng wrappers
def clean_cols(df):
    return DataEng.clean_cols(df)


def compute_product_series(df, tag_col="tag", var_col="variation"):
    return DataEng.compute_product_series(df, tag_col, var_col)


def nearest_row_at_price(prod_df, price, price_col="asp"):
    return DataEng.nearest_row_at_price(prod_df, price, price_col)


def pick_best_by_group(df, group_col, value_col):
    return DataEng.pick_best_by_group(df, group_col, value_col)


# opps wrappers
def ensure_days_valid_column(df):
    return OppsTable.ensure_days_valid_column(df)


def build_opp_table_columns(df):
    return OppsTable.build_columns(df)


# home ui wrappers
def home_step_card(title, desc):
    return HomeUI.step_card(title, desc)


def info_card(title, text):
    return HomeUI.info_card(title, text)


def build_footer_two_lines(
    signature_handle="@aqxiao",
    link_text="github.com/angie-xiao",
    link_href="https://github.com/angie-xiao",
):
    return HomeUI.footer_two_lines(signature_handle, link_text, link_href)


# overview ui wrappers
def kpi_card(id_title, title, id_value, bg="#f3f0f0", id_subtext=None):
    return OverviewUI.kpi_card(id_title, title, id_value, bg, id_subtext)


# formatter wrappers
def fmt_money(x):
    return Formatters.money(x)


def fmt_units(x):
    return Formatters.units(x)


def fmt_signed_units(x):
    return Formatters.signed_units(x)


def fmt_signed_money(x):
    return Formatters.signed_money(x)


def fmt_date(dt):
    return Formatters.date(dt)


# metrics / scenario / notes wrappers
def model_fit_units(prod_df):
    return Metrics.model_fit_units(prod_df)


def update_elasticity_kpi_by_product(product_name, elast_df):
    return Metrics.update_elasticity_kpi_by_product(product_name, elast_df)


def scenario_table(prod_df):
    return Scenario.table(prod_df)


def annualized_kpis_signed(asin, best50_df, curr_price_df, all_gam):
    return Scenario.annualized_kpis_signed(asin, best50_df, curr_price_df, all_gam)


def robustness_badge(prod_df):
    return Badges.robustness(prod_df)


def coverage_note(prod_df):
    return Notes.coverage(prod_df)
