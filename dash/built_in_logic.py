# pred graph - now p50, p975 are way too high
# curr price = 0
# annualized comparisons for shipped units & rev

# outliers
# shouldn't get tooo too much weight. for ex., while $80 asp might be valuable/rare, the fact that there's only been 1 purchase is not helpful
# need to take this into account & make it proportional too. maybe CVR???


# --------- built_in_logic.py (RMSE-focused; Top-N only; adds annualized opps & data range) ---------
import os, random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygam import ExpectileGAM, s
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.graph_objects as go


# --------------------------- Data engineering ---------------------------
class DataEngineer:
    def __init__(self, pricing_df, product_df, top_n=10):
        self.pricing_df = pricing_df
        self.product_df = product_df
        self.top_n = top_n

    def prepare(self) -> pd.DataFrame:
        
        def _clean_cols(df):
            df = df.copy()
            df.columns = df.columns.str.strip().str.lower()
            return df

        self.pricing_df = _clean_cols(self.pricing_df)
        self.product_df = _clean_cols(self.product_df)

        df = self.pricing_df.merge(self.product_df, how="left", on="asin")


        df["product"] = (df["tag"] + " " + df["weight"].astype(str)).str.upper()
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

        # ---- daily aggregation per product_key, then derive ASP ----
        daily = (
            df.groupby(["asin", "product", "order_date"])[
                ["shipped_units", "revenue_share_amt"]
            ]
            .sum()
        )
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
                daily_units=("shipped_units", lambda x: x.sum() / pd.Series(x.index).count()),  # add daily rate
                daily_rev=("revenue_share_amt", lambda x: x.sum() / pd.Series(x.index).count())  # add daily rate
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
            # & (asp_product_df["event_name"] == "NO DEAL")
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
            np.log(np.maximum(elasticity["asp_max"], eps)) -
            np.log(np.maximum(elasticity["asp_min"], eps))
        )
        elasticity["pct_change_qty"] = 100.0 * (
            np.log(np.maximum(elasticity["shipped_units_max"], eps)) -
            np.log(np.maximum(elasticity["shipped_units_min"], eps))
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
    
    Two-term ExpectileGAM:
      units ~ s(price, order=3) + s(days_sold, order=2)
    """
    def __init__(self, expectile=0.5, lam_grid=None, n_splines_grid=None):
        self.expectile = expectile
        self.lam_grid = np.array(lam_grid) if lam_grid is not None else np.logspace(-4, 3, 8)
        self.n_splines_grid = n_splines_grid or [5, 10, 20, 30]

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        scaler_X = StandardScaler()
        Xs = scaler_X.fit_transform(X.astype(float))
        terms = s(0, spline_order=3) + s(1, spline_order=2)

        gam = ExpectileGAM(terms, expectile=self.expectile)
        try:
            gam = gam.gridsearch(
                Xs, y.astype(float),
                lam=self.lam_grid,
                n_splines=self.n_splines_grid,
                weights=sample_weight
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
    """ (weights = reaction-per-day * tail) """
    def __init__(self, topsellers: pd.DataFrame,
                 gamma_time: float = 0.7,  # inverse-exp strength for days_sold
                 tail_strength: float = 0.6,
                 tail_p: float = 1.0):
        self.topsellers = topsellers
        self.gamma_time = float(gamma_time)
        self.tail_strength = float(tail_strength)
        self.tail_p = float(tail_p)
        
    def _make_weights(self, sub: pd.DataFrame) -> np.ndarray:
        """Weights based on daily revenue and days_sold for confidence in the data point"""
        # Base weights from daily revenue (more weight to high-revenue price points)
        daily_rev = np.clip(sub["daily_rev"].values.astype(float), 1.0, None)
        w = np.sqrt(daily_rev)

        # Time decay (more weight to price points with more days of data)
        days = np.clip(sub["days_sold"].values.astype(float), 1.0, None)
        d_med = np.median(days) if np.median(days) > 0 else 1.0
        w_time = np.exp(-self.gamma_time * (days / d_med))
        w *= w_time

        # Price range weighting (more weight to prices with more data)
        if self.tail_strength > 0:
            asp = sub["asp"].values.astype(float)
            q25, q75 = np.percentile(asp, [25, 75])
            iqr = q75 - q25 if q75 > q25 else 1.0
            
            # Calculate distance from median in terms of IQR
            rel_dist = np.abs(asp - np.median(asp)) / iqr
            
            # Apply tail strength but cap the maximum weight multiplier
            w *= (1.0 + self.tail_strength * np.minimum(rel_dist, 2.0)**self.tail_p)

        # Normalize weights 
        w = w / np.mean(w) if w.size > 0 else w
        return w

    def run(self) -> pd.DataFrame:
        all_results = []
        for product, sub in self.topsellers.groupby("product"):
            sub = sub.sort_values("asp").reset_index(drop=True)
            
            # Features: price and days_sold
            X = sub[["asp", "days_sold"]].to_numpy(dtype=float)
            # Target: daily revenue
            y = sub["daily_rev"].to_numpy(dtype=float)
            
            # Weights based on daily revenue and days_sold
            weights = self._make_weights(sub)  

            out = {}
            for q in [0.025, 0.5, 0.975]:
                gam = GAMTuner(expectile=q).fit(X, y, sample_weight=weights)
                y_pred = gam.predict(gam._scaler_X.transform(X))
                # Add prefix 'revenue_' since these are revenue predictions
                out[f"revenue_pred_{q}"] = y_pred

            preds = pd.DataFrame(out, index=sub.index)
            
            # Add raw predictions without revenue prefix for compatibility
            for q in [0.025, 0.5, 0.975]:
                preds[f"pred_{q}"] = preds[f"revenue_pred_{q}"]
            
            results = pd.concat(
                [sub[["asp", "days_sold", "product", "asin", "shipped_units", "daily_units", "daily_rev"]].reset_index(drop=True),
                preds.reset_index(drop=True)],
                axis=1
            )
            all_results.append(results)

        all_gam_results = pd.concat(all_results, axis=0, ignore_index=True)

        # Add revenue_pred_avg
        rev_cols = [c for c in all_gam_results.columns if c.startswith("revenue_pred_")]
        all_gam_results["revenue_pred_avg"] = all_gam_results[rev_cols].mean(axis=1)
        
        # Set revenue_actual
        all_gam_results["revenue_actual"] = all_gam_results["daily_rev"]

        all_gam_results["asin"] = all_gam_results["asin"].astype(str)
        
        return all_gam_results


# --------------------------- Optimizer (plain) ---------------------------
class Optimizer:
    @staticmethod
    def run(all_gam_results: pd.DataFrame) -> dict:
        def pick_best(col):
            if col not in all_gam_results or all_gam_results[col].isna().all():
                return pd.DataFrame()
            return all_gam_results.loc[all_gam_results.groupby("product")[col].idxmax()]
        return {
            "best_avg": pick_best("revenue_pred_avg"),
            "best50":   pick_best("revenue_pred_0.5"),
            "best975":  pick_best("revenue_pred_0.975"),
            "best25":   pick_best("revenue_pred_0.025"),
            "all_gam_results": all_gam_results,
        }


# --------------------------- Pipeline ---------------------------
class PricingPipeline:
    def __init__(self, pricing_df, product_df, top_n=10):
        self.engineer = DataEngineer(pricing_df, product_df, top_n)

    def _build_curr_price_df(self) -> pd.DataFrame:
        """ current price df with product tag """
        product = self.engineer.product_df.copy()
        product["product"] = (product["tag"] + " " + product["weight"].astype(str)).str.upper() 
        out = product[['asin','product','current_price']]
        return out.reset_index(drop=True)


    @staticmethod
    def _nearest_row_at_price(prod_df: pd.DataFrame, price: float):
        if prod_df.empty or pd.isna(price):
            return None
        idx = (prod_df["asp"] - price).abs().idxmin()
        try:
            return prod_df.loc[idx]
        except Exception:
            return None

    def assemble_dashboard_frames(self) -> dict:
        # 1) core tables
        topsellers = self.engineer.prepare()                  # Top-N only
        elasticity_df = ElasticityAnalyzer.compute(topsellers)  # UI only
        all_gam_results = GAMModeler(topsellers, tail_strength=0.6, tail_p=1.0).run()

        # 2) optimizer + best tables
        bests   = Optimizer.run(all_gam_results)
        best_avg = bests["best_avg"].copy()

        if "asin" not in best_avg.columns:
            pk_map = topsellers[["product", "asin"]].drop_duplicates()
            best_avg = best_avg.merge(pk_map, on="product", how="left")

        if "revenue_actual" not in best_avg.columns:
            if {"asp", "shipped_units"}.issubset(best_avg.columns):
                best_avg["revenue_actual"] = best_avg["asp"] * best_avg["shipped_units"]
            else:
                ra = (all_gam_results[["product", "asp", "revenue_actual"]].drop_duplicates())
                best_avg = best_avg.merge(ra, on=["product", "asp"], how="left")

        # 3) current prices
        curr_price_df = self._build_curr_price_df()
        # print(f"Current price DataFrame shape: {curr_price_df.shape}")
        # print("Sample of current prices:")
        # print(curr_price_df.head())

        # 4) data range + annualization factor
        df_dates = self.engineer.pricing_df.copy()
        df_dates["order_date"] = pd.to_datetime(df_dates["order_date"], errors="coerce")
        data_start = pd.to_datetime(df_dates["order_date"].min()) if len(df_dates) else pd.NaT
        data_end   = pd.to_datetime(df_dates["order_date"].max()) if len(df_dates) else pd.NaT
        days_covered = int((data_end - data_start).days) + 1 if pd.notna(data_start) and pd.notna(data_end) else 0
        annual_factor = (365.0 / max(1, days_covered)) if days_covered else 1.0

        # 5) best-50 by expected revenue
        if "revenue_pred_0.5" in all_gam_results.columns:
            idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
            best50 = (
                all_gam_results.loc[idx, ["product", "asin", "asp", "pred_0.5", "revenue_pred_0.5"]]
                .drop_duplicates(subset=["product"])
                .reset_index(drop=True)
            )
        else:
            best50 = pd.DataFrame(columns=["product","asin","asp","pred_0.5","revenue_pred_0.5"])

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

            curr_row = self._nearest_row_at_price(prod_curve, curr_price) if pd.notna(curr_price) else None
            if curr_row is not None:
                units_curr = float(curr_row.get("pred_0.5", np.nan))
                rev_curr   = float(curr_row.get("revenue_pred_0.5", np.nan))
            else:
                units_curr, rev_curr = np.nan, np.nan

            du = (units_best - units_curr) if (pd.notna(units_best) and pd.notna(units_curr)) else np.nan
            dr = (rev_best - rev_curr)     if (pd.notna(rev_best) and pd.notna(rev_curr))   else np.nan

            rows.append({
                "product": p, "asin": pk,
                "current_price": curr_price, "best_price": asp_best,
                "units_pred_best": units_best, "units_pred_curr": units_curr,
                "revenue_pred_best": rev_best, "revenue_pred_curr": rev_curr,
                "delta_units": du, "delta_revenue": dr,
                "delta_units_annual": du * annual_factor if pd.notna(du) else np.nan,
                "delta_revenue_annual": dr * annual_factor if pd.notna(dr) else np.nan,
                "revenue_best_annual": rev_best * annual_factor if pd.notna(rev_best) else np.nan,
            })
        opps_summary = pd.DataFrame(rows)

        # 7) normalize key dtype across frames
        for df in (best_avg, all_gam_results, curr_price_df, topsellers, opps_summary):
            if "asin" in df.columns:
                df["asin"] = df["asin"].astype(str)

        # 8) frames dict
        frames = {
            "price_quant_df": (topsellers.groupby(["asp", "product"])["shipped_units"].sum().reset_index()),
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
    def from_csv_folder(cls, base_dir, data_folder="data",
                        pricing_file="730d.csv", product_file="products.csv", top_n=10):
        pricing_df = pd.read_csv(os.path.join(base_dir, data_folder, pricing_file))
        product_df = pd.read_csv(os.path.join(base_dir, data_folder, product_file))
        return cls(pricing_df, product_df, top_n).assemble_dashboard_frames()


# --------------------------- viz ---------------------------
class viz:
    def __init__(self, template="lux"):
        templates = ["bootstrap","minty","pulse","flatly","quartz","cyborg","darkly","vapor","lux"]
        load_figure_template(templates)
        self.template = template

    def gam_results(self, all_gam_results: pd.DataFrame):
        
        ''' '''
        product_lst = all_gam_results["product"].unique()
        pltly_qual = px.colors.qualitative.Dark24
        pltly_qual.extend(px.colors.qualitative.Vivid)
        colors = random.sample(pltly_qual, len(product_lst))
        color_dct = {p: colors[i] for i, p in enumerate(product_lst)}

        fig = go.Figure()
        for group_name, group_df in all_gam_results.groupby("product"):
            best_50 = group_df.loc[group_df["revenue_pred_0.5"].idxmax()].to_frame().T
            best_025 = group_df.loc[group_df["revenue_pred_0.025"].idxmax()].to_frame().T
            best_975 = group_df.loc[group_df["revenue_pred_0.975"].idxmax()].to_frame().T

            fig.add_trace(go.Scatter(
                name=f"group {group_name} band",
                x=group_df["asp"].tolist() + group_df["asp"].tolist()[::-1],
                y=group_df["revenue_pred_0.975"].tolist() + group_df["revenue_pred_0.025"].tolist()[::-1],
                fill="toself", fillcolor="#cbcbcb",
                line=dict(color="#cbcbcb"), legendgroup=group_name,
                showlegend=False, opacity=0.4,
            ))
            fig.add_trace(go.Scatter(
                x=group_df["asp"], y=group_df["revenue_actual"],
                mode="markers", name="Revenue Actual",
                marker=dict(symbol="x", color=color_dct[group_name], size=10),
                opacity=0.5,
            ))
            fig.add_trace(go.Scatter(
                x=best_50["asp"], y=best_50["revenue_pred_0.5"],
                mode="markers", marker=dict(color="#B82132", size=18),
                name="Rec. Price",
            ))
            fig.add_trace(go.Scatter(
                x=group_df["asp"], y=group_df["revenue_pred_0.5"],
                mode="lines", marker=dict(color="#B82132"),
                name="Expected Revenue Prediction",
            ))
            fig.add_trace(go.Scatter(
                x=best_025["asp"], y=best_025["revenue_pred_0.025"],
                mode="markers", marker=dict(color="#AAB396", size=18),
                name="Conservative Prediction",
            ))
            fig.add_trace(go.Scatter(
                x=best_975["asp"], y=best_975["revenue_pred_0.975"],
                mode="markers", marker=dict(color="#F2B28C", size=18),
                name="Optimistic Prediction",
            ))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
                          margin=dict(r=8))
        return fig

    def elast_dist(self, elast_df: pd.DataFrame):
        fig = (px.histogram(elast_df, x="ratio", width=1200, height=600, template=self.template)
               .update_xaxes(title_text="Elasticity")
               .update_yaxes(title_text="Product Count"))
        return fig

    def opportunity_chart(elast_df, best50_df, curr_df, all_gam):
        import plotly.express as px
        for df in [elast_df, best50_df, curr_df, all_gam]:
            if df is None or getattr(df, "empty", True):
                return {}
        if ("revenue_pred_0.5" not in all_gam.columns) or ("revenue_pred_0.5" not in best50_df.columns):
            return {}
        prods = sorted(set(all_gam["product"]) & set(best50_df["product"]) & set(curr_df["product"]))
        if not prods:
            return {}
        rows = []
        for p in prods:
            try:
                curr_price = curr_df.loc[curr_df["product"] == p, "current_price"]
                if curr_price.empty or pd.isna(curr_price.iloc[0]):
                    continue
                curr_price = float(curr_price.iloc[0])
                prod = all_gam[(all_gam["product"] == p) & pd.notna(all_gam["asp"]) & pd.notna(all_gam["revenue_pred_0.5"])]
                if prod.empty:
                    continue
                idx = (prod["asp"] - curr_price).abs().idxmin()
                rev_curr = float(prod.loc[idx, "revenue_pred_0.5"])
                rec = best50_df.loc[best50_df["product"] == p]
                if rec.empty:
                    continue
                rev_best = float(rec["revenue_pred_0.5"].iloc[0])
                upside = rev_best - rev_curr
                e = elast_df.loc[elast_df["product"] == p, "ratio"]
                elast_val = float(e.iloc[0]) if len(e) else np.nan
                rows.append({"product": p, "upside": upside, "elasticity": elast_val})
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if df.empty:
            return {}
        df = df.sort_values("upside", ascending=False).head(12)
        fig = px.bar(df, x="product", y="upside", hover_data=["elasticity"], height=380)
        fig.update_yaxes(title_text="Upside (Expected Revenue Δ)", tickprefix="$", separatethousands=True)
        fig.update_traces(text=df["upside"].map(lambda x: f"${x:,.0f}"), textposition="outside", cliponaxis=False)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=60), uniformtext_minsize=10, uniformtext_mode="hide",
                          yaxis={"categoryorder": "total descending"})
        return fig

    def empty_fig(self, title="No data"):
        fig = go.Figure()
        fig.update_layout(title=title, template=self.template, xaxis=dict(visible=False), yaxis=dict(visible=False),
                          margin=dict(l=10, r=10, t=60, b=40))
        fig.add_annotation(text=title, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        return fig



if __name__ == "__main__":
    pricing_df, product_df = pd.read_csv('data/730d.csv'), pd.read_csv('data/products.csv')
    
    # DataEngineer(pricing_df, product_df, top_n=10).prepare()
    
    # GAMModeler(
    #     DataEngineer(pricing_df, product_df, top_n=10).prepare(),).run()
    
    PricingPipeline(pricing_df, product_df, top_n=10)._build_curr_price_df()