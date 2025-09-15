# --------- built_in_logic.py (RMSE-focused; Top-N only; adds annualized opps & data range) ---------
import os, random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygam import ExpectileGAM, s
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

# local import
from helpers import (
    clean_cols,
    compute_product_series,
    nearest_row_at_price,
    pick_best_by_group,
)


# --------------------------- Data engineering ---------------------------
class DataEngineer:
    def __init__(self, pricing_df, product_df, top_n=10):
        self.pricing_df = pricing_df
        self.product_df = product_df
        self.top_n = top_n

    def prepare(self) -> pd.DataFrame:
        # normalize columns
        self.pricing_df = clean_cols(self.pricing_df)
        self.product_df = clean_cols(self.product_df)

        df = self.pricing_df.merge(self.product_df, how="left", on="asin")

        # canonical product label
        df["product"] = compute_product_series(df)

        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

        # ---- daily aggregation per asin, then derive ASP ----
        daily = df.groupby(["asin", "product", "order_date"])[
            ["shipped_units", "revenue_share_amt"]
        ].sum()
        daily = daily[daily["shipped_units"] > 0]
        daily["asp"] = daily["revenue_share_amt"] / daily["shipped_units"]
        daily = daily[daily["asp"] > 0]
        daily["asp"] = daily["asp"].round(1)
        daily.reset_index(inplace=True)

        # ---- product × ASP table with daily rates and totals ----
        asp_product_df = (
            daily.groupby(["asin", "product", "asp"])
            .agg(
                shipped_units=("shipped_units", "sum"),  # keep total units
                revenue_share_amt=("revenue_share_amt", "sum"),
                days_sold=("order_date", "count"),
                daily_units=(
                    "shipped_units",
                    lambda x: x.sum() / pd.Series(x.index).count(),
                ),
                daily_rev=(
                    "revenue_share_amt",
                    lambda x: x.sum() / pd.Series(x.index).count(),
                ),
            )
            .reset_index()
        )

        # Top-N products by total revenue — KEEP ONLY those products (no over-display)
        top_n_products = (
            asp_product_df.groupby("product")["revenue_share_amt"]
            .sum()
            .reset_index()
            .sort_values("revenue_share_amt", ascending=False)["product"]
            .head(self.top_n)
            .tolist()
        )

        filtered = asp_product_df[
            (asp_product_df["product"].isin(top_n_products))
        ].copy()

        # Normalize dtype for downstream joins
        filtered["asin"] = filtered["asin"].astype(str)
        return filtered


# --------------------------- Elasticity (UI summary only) ---------------------------
class ElasticityAnalyzer:
    @staticmethod
    def compute(topsellers: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-9
        elasticity = (
            topsellers.groupby("product")
            .agg(
                asp_max=("asp", "max"),
                asp_min=("asp", "min"),
                shipped_units_max=("shipped_units", "max"),
                shipped_units_min=("shipped_units", "min"),
                product_count=("product", "count"),
            )
            .reset_index()
        )
        elasticity["pct_change_price"] = 100.0 * (
            np.log(np.maximum(elasticity["asp_max"], eps))
            - np.log(np.maximum(elasticity["asp_min"], eps))
        )
        elasticity["pct_change_qty"] = 100.0 * (
            np.log(np.maximum(elasticity["shipped_units_max"], eps))
            - np.log(np.maximum(elasticity["shipped_units_min"], eps))
        )
        elasticity["ratio"] = elasticity["pct_change_qty"] / np.where(
            elasticity["pct_change_price"] == 0, np.nan, elasticity["pct_change_price"]
        )
        elasticity["pct"] = elasticity["ratio"].rank(pct=True) * 100
        return elasticity.sort_values("ratio", ascending=False).reset_index(drop=True)


# --------------------------- GAM ---------------------------
class GAMTuner:
    """
    (auto-gridsearch; raw y)
    Two-term ExpectileGAM:      units ~ s(price, order=3) + s(days_sold, order=2)
    """

    def __init__(self, expectile=0.5, lam_grid=None, n_splines_grid=None):
        self.expectile = expectile
        self.lam_grid = (
            np.array(lam_grid) if lam_grid is not None else np.logspace(-4, 3, 8)
        )
        self.n_splines_grid = n_splines_grid or [5, 10, 20, 30]

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        scaler_X = StandardScaler()
        Xs = scaler_X.fit_transform(X.astype(float))
        terms = s(0, spline_order=3) + s(1, spline_order=2)
        gam = ExpectileGAM(terms, expectile=self.expectile)
        try:
            gam = gam.gridsearch(
                Xs,
                y.astype(float),
                lam=self.lam_grid,
                n_splines=self.n_splines_grid,
                weights=sample_weight,
            )
        except Exception as e:
            print(f"Gridsearch failed; fallback: {e}")
            gam = ExpectileGAM(terms, expectile=self.expectile).fit(
                Xs, y.astype(float), weights=sample_weight
            )
        gam._scaler_X = scaler_X
        return gam


# --------------------------- Modeling  ---------------------------
class GAMModeler:
    """(weights = reaction-per-day * tail)"""

    def __init__(
        self,
        topsellers: pd.DataFrame,
        gamma_time: float = 0.7,  # inverse-exp strength for days_sold
        tail_strength: float = 0.6,
        tail_p: float = 1.0,
    ):
        self.topsellers = topsellers
        self.gamma_time = float(gamma_time)
        self.tail_strength = float(tail_strength)
        self.tail_p = float(tail_p)

    def _make_weights(self, sub: pd.DataFrame) -> np.ndarray:
        """Weights based on daily revenue and days_sold for confidence in the data point"""
        daily_rev = np.clip(sub["daily_rev"].values.astype(float), 1.0, None)
        w = np.sqrt(daily_rev)
        days = np.clip(sub["days_sold"].values.astype(float), 1.0, None)
        d_med = np.median(days) if np.median(days) > 0 else 1.0
        w_time = np.exp(-self.gamma_time * (days / d_med))
        w *= w_time
        if self.tail_strength > 0:
            asp = sub["asp"].values.astype(float)
            q25, q75 = np.percentile(asp, [25, 75])
            iqr = q75 - q25 if q75 > q25 else 1.0
            rel_dist = np.abs(asp - np.median(asp)) / iqr
            w *= 1.0 + self.tail_strength * np.minimum(rel_dist, 2.0) ** self.tail_p
        w = w / np.mean(w) if w.size > 0 else w
        return w

    def run(self) -> pd.DataFrame:
        all_results = []
        for product, sub in self.topsellers.groupby("product"):
            sub = sub.sort_values("asp").reset_index(drop=True)
            X = sub[["asp", "days_sold"]].to_numpy(dtype=float)
            y = sub["daily_rev"].to_numpy(dtype=float)
            weights = self._make_weights(sub)
            out = {}
            for q in [0.025, 0.5, 0.975]:
                gam = GAMTuner(expectile=q).fit(X, y, sample_weight=weights)
                y_pred = gam.predict(gam._scaler_X.transform(X))
                out[f"revenue_pred_{q}"] = y_pred
            preds = pd.DataFrame(out, index=sub.index)
            for q in [0.025, 0.5, 0.975]:
                preds[f"pred_{q}"] = preds[f"revenue_pred_{q}"]
            results = pd.concat(
                [
                    sub[
                        [
                            "asp",
                            "days_sold",
                            "product",
                            "asin",
                            "shipped_units",
                            "daily_units",
                            "daily_rev",
                        ]
                    ].reset_index(drop=True),
                    preds.reset_index(drop=True),
                ],
                axis=1,
            )
            all_results.append(results)

        all_gam_results = pd.concat(all_results, axis=0, ignore_index=True)
        rev_cols = [c for c in all_gam_results.columns if c.startswith("revenue_pred_")]
        all_gam_results["revenue_pred_avg"] = all_gam_results[rev_cols].mean(axis=1)
        all_gam_results["revenue_actual"] = all_gam_results["daily_rev"]
        all_gam_results["asin"] = all_gam_results["asin"].astype(str)
        return all_gam_results


# --------------------------- Optimizer (plain) ---------------------------
class Optimizer:
    @staticmethod
    def run(all_gam_results: pd.DataFrame) -> dict:
        return {
            "best_avg": pick_best_by_group(
                all_gam_results, "product", "revenue_pred_avg"
            ),
            "best50": pick_best_by_group(
                all_gam_results, "product", "revenue_pred_0.5"
            ),
            "best975": pick_best_by_group(
                all_gam_results, "product", "revenue_pred_0.975"
            ),
            "best25": pick_best_by_group(
                all_gam_results, "product", "revenue_pred_0.025"
            ),
            "all_gam_results": all_gam_results,
        }


# --------------------------- Pipeline ---------------------------
class PricingPipeline:
    def __init__(self, pricing_df, product_df, top_n=10):
        self.engineer = DataEngineer(pricing_df, product_df, top_n)

    def _build_curr_price_df(self) -> pd.DataFrame:
        """current price df with product tag"""
        product = self.engineer.product_df.copy()
        product["product"] = compute_product_series(product)
        out = product[["asin", "product", "current_price"]]
        return out.reset_index(drop=True)

    def assemble_dashboard_frames(self) -> dict:
        # 1) core tables
        topsellers = self.engineer.prepare()  # Top-N only
        elasticity_df = ElasticityAnalyzer.compute(topsellers)  # UI only
        all_gam_results = GAMModeler(topsellers, tail_strength=0.6, tail_p=1.0).run()

        # 2) optimizer + best tables
        bests = Optimizer.run(all_gam_results)
        best_avg = bests["best_avg"].copy()

        if "asin" not in best_avg.columns:
            pk_map = topsellers[["product", "asin"]].drop_duplicates()
            best_avg = best_avg.merge(pk_map, on="product", how="left")

        if "revenue_actual" not in best_avg.columns:
            if {"asp", "shipped_units"}.issubset(best_avg.columns):
                best_avg["revenue_actual"] = best_avg["asp"] * best_avg["shipped_units"]
            else:
                ra = all_gam_results[
                    ["product", "asp", "revenue_actual"]
                ].drop_duplicates()
                best_avg = best_avg.merge(ra, on=["product", "asp"], how="left")

        # 3) current prices
        curr_price_df = self._build_curr_price_df()

        # 4) data range + annualization factor
        df_dates = self.engineer.pricing_df.copy()
        df_dates["order_date"] = pd.to_datetime(df_dates["order_date"], errors="coerce")
        data_start = (
            pd.to_datetime(df_dates["order_date"].min()) if len(df_dates) else pd.NaT
        )
        data_end = (
            pd.to_datetime(df_dates["order_date"].max()) if len(df_dates) else pd.NaT
        )
        days_covered = (
            int((data_end - data_start).days) + 1
            if pd.notna(data_start) and pd.notna(data_end)
            else 0
        )
        annual_factor = (365.0 / max(1, days_covered)) if days_covered else 1.0

        # 5) best-50 by expected revenue
        if "revenue_pred_0.5" in all_gam_results.columns:
            idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
            best50 = (
                all_gam_results.loc[
                    idx, ["product", "asin", "asp", "pred_0.5", "revenue_pred_0.5"]
                ]
                .drop_duplicates(subset=["product"])
                .reset_index(drop=True)
            )
        else:
            best50 = pd.DataFrame(
                columns=["product", "asin", "asp", "pred_0.5", "revenue_pred_0.5"]
            )

        # 6) opportunity summary per product (units & revenue, + annualized)
        rows = []
        for _, r in best50.iterrows():
            p = r["product"]
            pk = str(r["asin"])
            asp_best = float(r["asp"])
            units_best = float(r.get("pred_0.5", np.nan))
            rev_best = float(r.get("revenue_pred_0.5", np.nan))

            prod_curve = all_gam_results[(all_gam_results["product"] == p)]
            cp_ser = curr_price_df.loc[curr_price_df["asin"] == pk, "current_price"]
            curr_price = float(cp_ser.iloc[0]) if len(cp_ser) else np.nan

            curr_row = (
                nearest_row_at_price(prod_curve, curr_price)
                if pd.notna(curr_price)
                else None
            )
            if curr_row is not None:
                units_curr = float(curr_row.get("pred_0.5", np.nan))
                rev_curr = float(curr_row.get("revenue_pred_0.5", np.nan))
            else:
                units_curr, rev_curr = np.nan, np.nan

            du = (
                (units_best - units_curr)
                if (pd.notna(units_best) and pd.notna(units_curr))
                else np.nan
            )
            dr = (
                (rev_best - rev_curr)
                if (pd.notna(rev_best) and pd.notna(rev_curr))
                else np.nan
            )

            rows.append(
                {
                    "product": p,
                    "asin": pk,
                    "current_price": curr_price,
                    "best_price": asp_best,
                    "units_pred_best": units_best,
                    "units_pred_curr": units_curr,
                    "revenue_pred_best": rev_best,
                    "revenue_pred_curr": rev_curr,
                    "delta_units": du,
                    "delta_revenue": dr,
                    "delta_units_annual": (
                        du * annual_factor if pd.notna(du) else np.nan
                    ),
                    "delta_revenue_annual": (
                        dr * annual_factor if pd.notna(dr) else np.nan
                    ),
                    "revenue_best_annual": (
                        rev_best * annual_factor if pd.notna(rev_best) else np.nan
                    ),
                }
            )

        opps_summary = pd.DataFrame(rows)

        # 7) normalize key dtype across frames
        for df in (best_avg, all_gam_results, curr_price_df, topsellers, opps_summary):
            if "asin" in df.columns:
                df["asin"] = df["asin"].astype(str)

        # 8) frames dict
        frames = {
            "price_quant_df": (
                topsellers.groupby(["asp", "product"])["shipped_units"]
                .sum()
                .reset_index()
            ),
            "best_avg": best_avg,
            "all_gam_results": all_gam_results,
            "best_optimal_pricing_df": best_avg[
                ["product", "asin", "asp", "revenue_pred_avg", "revenue_actual"]
            ].copy(),
            "elasticity_df": elasticity_df[["product", "ratio", "pct"]],
            "curr_opt_df": best_avg,
            "curr_price_df": curr_price_df,
            "opps_summary": opps_summary,
            "meta": {
                "data_start": data_start,
                "data_end": data_end,
                "days_covered": days_covered,
                "annual_factor": annual_factor,
            },
        }
        return frames

    @classmethod
    def from_csv_folder(
        cls,
        base_dir,
        data_folder="data",
        pricing_file="pricing.csv",
        product_file="products.csv",
        top_n=10,
    ):
        pricing_df = pd.read_csv(os.path.join(base_dir, data_folder, pricing_file))
        product_df = pd.read_csv(os.path.join(base_dir, data_folder, product_file))
        return cls(pricing_df, product_df, top_n).assemble_dashboard_frames()


# --------------------------- viz ---------------------------
class viz:
    def __init__(self, template="lux"):
        templates = [
            "bootstrap",
            "minty",
            "pulse",
            "flatly",
            "quartz",
            "cyborg",
            "darkly",
            "vapor",
            "lux",
        ]
        load_figure_template(templates)
        self.template = template

    def gam_results(self, all_gam_results: pd.DataFrame):
        """ """
        product_lst = all_gam_results["product"].unique()
        pltly_qual = px.colors.qualitative.Dark24
        pltly_qual.extend(px.colors.qualitative.Vivid)
        colors = random.sample(pltly_qual, len(product_lst))
        color_dct = {p: colors[i] for i, p in enumerate(product_lst)}

        fig = go.Figure()
        for group_name, group_df in all_gam_results.groupby("product"):
            best_50 = group_df.loc[group_df["revenue_pred_0.5"].idxmax()].to_frame().T
            best_025 = (
                group_df.loc[group_df["revenue_pred_0.025"].idxmax()].to_frame().T
            )
            best_975 = (
                group_df.loc[group_df["revenue_pred_0.975"].idxmax()].to_frame().T
            )

            fig.add_trace(
                go.Scatter(
                    name=f"group {group_name} band",
                    x=group_df["asp"].tolist() + group_df["asp"].tolist()[::-1],
                    y=group_df["revenue_pred_0.975"].tolist()
                    + group_df["revenue_pred_0.025"].tolist()[::-1],
                    fill="toself",
                    fillcolor="#cbcbcb",
                    line=dict(color="#cbcbcb"),
                    legendgroup=group_name,
                    showlegend=False,
                    opacity=0.4,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=group_df["asp"],
                    y=group_df["revenue_actual"],
                    mode="markers",
                    name="Revenue Actual",
                    marker=dict(symbol="x", color=color_dct[group_name], size=10),
                    opacity=0.5,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=best_50["asp"],
                    y=best_50["revenue_pred_0.5"],
                    mode="markers",
                    marker=dict(color="#B82132", size=18),
                    name="Rec. Price",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=group_df["asp"],
                    y=group_df["revenue_pred_0.5"],
                    mode="lines",
                    marker=dict(color="#B82132"),
                    name="Expected Revenue Prediction",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=best_025["asp"],
                    y=best_025["revenue_pred_0.025"],
                    mode="markers",
                    marker=dict(color="#AAB396", size=18),
                    name="Conservative Prediction",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=best_975["asp"],
                    y=best_975["revenue_pred_0.975"],
                    mode="markers",
                    marker=dict(color="#F2B28C", size=18),
                    name="Optimistic Prediction",
                )
            )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            margin=dict(r=8),
        )

        fig.update_xaxes(
            title_text="Average Selling Price (ASP)",
            tickprefix="$",
            separatethousands=True,
        )
        fig.update_yaxes(
            title_text="Expected Daily Revenue", tickprefix="$", separatethousands=True
        )

        return fig

    def elast_dist(self, elast_df: pd.DataFrame):
        fig = (
            px.histogram(
                elast_df, x="ratio", width=1200, height=600, template=self.template
            )
            .update_xaxes(title_text="Elasticity")
            .update_yaxes(title_text="Product Count")
        )
        return fig

    def opportunity_chart(self, elast_df, best50_df, curr_df, all_gam):
        """
        Build the Top-N upside bar chart (Expected Revenue Δ at recommended vs current price).
        Aligns by product, uses P50 revenue, and annotates with elasticity.
        """
        # basic guards
        for df in (elast_df, best50_df, curr_df, all_gam):
            if df is None or getattr(df, "empty", True):
                return self.empty_fig("No data for opportunity chart")

        # required columns
        need_cols = [
            (all_gam, ["product", "asp", "revenue_pred_0.5"]),
            (best50_df, ["product", "asp", "revenue_pred_0.5"]),
            (curr_df, ["product", "current_price"]),
            (elast_df, ["product", "ratio"]),
        ]
        for frame, cols in need_cols:
            for c in cols:
                if c not in frame.columns:
                    return self.empty_fig(f"Missing column: {c}")

        # numeric coercions
        to_num = [
            (all_gam, ["asp", "revenue_pred_0.5"]),
            (best50_df, ["asp", "revenue_pred_0.5"]),
            (curr_df, ["current_price"]),
            (elast_df, ["ratio"]),
        ]
        for frame, cols in to_num:
            for c in cols:
                frame[c] = (
                    pd.to_numeric(frame[c], errors="coerce")
                    if c in frame.columns
                    else frame.get(c)
                )

        prods = sorted(
            set(all_gam["product"])
            & set(best50_df["product"])
            & set(curr_df["product"])
        )
        if not prods:
            return self.empty_fig("No overlapping products across inputs")

        rows = []
        for p in prods:
            try:
                cp_ser = curr_df.loc[curr_df["product"] == p, "current_price"]
                if cp_ser.empty or pd.isna(cp_ser.iloc[0]):
                    continue
                curr_price = float(cp_ser.iloc[0])

                prod_curve = all_gam[
                    (all_gam["product"] == p)
                    & pd.notna(all_gam["asp"])
                    & pd.notna(all_gam["revenue_pred_0.5"])
                ]
                if prod_curve.empty:
                    continue

                # current revenue ~ nearest modeled ASP
                idx = (prod_curve["asp"] - curr_price).abs().idxmin()
                rev_curr = float(prod_curve.loc[idx, "revenue_pred_0.5"])

                # recommended (from best50 rows)
                rec = best50_df.loc[best50_df["product"] == p]
                if rec.empty or pd.isna(rec["revenue_pred_0.5"].iloc[0]):
                    continue
                rev_best = float(rec["revenue_pred_0.5"].iloc[0])

                # elasticity (optional)
                e = elast_df.loc[elast_df["product"] == p, "ratio"]
                elast_val = float(e.iloc[0]) if len(e) and pd.notna(e.iloc[0]) else None

                rows.append(
                    {
                        "product": p,
                        "upside": rev_best - rev_curr,
                        "elasticity": elast_val,
                    }
                )
            except Exception:
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return self.empty_fig("No computable upside")

        # Top N largest upside; horizontal bar for readability
        df = df.sort_values("upside", ascending=True).tail(12)

        # plotting
        BEIGE = "#EDD4B7"
        # GOLD = "#DAA520"
        fig = px.bar(
            df,
            y="product",
            x="upside",
            hover_data=["elasticity"],
            height=420,
            template=self.template,
            color_discrete_sequence=[BEIGE],  # set bar color
        )
        fig.update_xaxes(
            title_text="Daily Expected Revenue Δ",
            tickprefix="$",
            separatethousands=True,
        )
        fig.update_yaxes(title_text="")

        # keep bars gold even if the template tries to restyle traces
        fig.update_traces(
            marker_color=BEIGE,
            text=df["upside"].map(lambda x: f"${x:,.0f}"),
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=60),
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )
        return fig

    def empty_fig(self, title="No data"):
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template=self.template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=60, b=40),
        )
        fig.add_annotation(
            text=title, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper"
        )
        return fig


# if __name__ == "__main__":
#     pricing_df, product_df = pd.read_csv('data/730d.csv'), pd.read_csv('data/products.csv')

#     # DataEngineer(pricing_df, product_df, top_n=10).prepare()

#     # GAMModeler(
#     #     DataEngineer(pricing_df, product_df, top_n=10).prepare(),).run()

#     PricingPipeline(pricing_df, product_df, top_n=10)._build_curr_price_df()
