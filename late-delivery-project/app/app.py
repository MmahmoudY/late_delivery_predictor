"""Late-Delivery Risk app — order risk checker + seller scorecard (Olist marketplace)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="Late-Delivery Risk", page_icon="📦", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_resource
def load_model():
    return joblib.load(ART / "late_delivery_model.joblib")


@st.cache_data
def load_tables():
    meta = json.load(open(ART / "metadata.json"))
    sellers = pd.read_csv(ART / "seller_stats.csv")
    routes = pd.read_csv(ART / "route_stats.csv").set_index("route")
    cats = pd.read_csv(ART / "category_stats.csv").set_index("main_category")
    locs = pd.read_csv(ART / "locations.csv")
    buckets = pd.read_csv(ART / "delay_bucket_satisfaction.csv")
    imp = pd.read_csv(ART / "feature_importance.csv")
    return meta, sellers, routes, cats, locs, buckets, imp


model = load_model()
meta, sellers, routes, cats, locs, buckets, imp = load_tables()
G = meta["global_late_rate"]
HIGH, MED = meta["threshold_high"], meta["threshold_medium"]


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


def city_coords(state, city):
    r = locs[(locs.state == state) & (locs.city == city)]
    return (float(r.lat.iloc[0]), float(r.lng.iloc[0])) if len(r) else (np.nan, np.nan)


def band_of(p):
    return ("High", "🔴") if p >= HIGH else ("Medium", "🟠") if p >= MED else ("Low", "🟢")


# ----------------------------------------------------------------------------- header
st.title("📦 Late-Delivery Risk")
st.caption("Predict at purchase time whether an order will arrive after its estimated delivery date — "
           "built on ~96K real orders from the Olist Brazilian marketplace.")

tab1, tab2, tab3, tab4 = st.tabs(["🧾 Order risk checker", "🏪 Seller scorecard", "⭐ Satisfaction impact", "ℹ️ About the model"])

# ----------------------------------------------------------------------------- tab 1
with tab1:
    st.subheader("Enter an order")
    st.caption("Seller reliability, route speed and current network conditions are looked up automatically "
               f"(latest data: {meta['as_of'][:10]}).")

    all_states = sorted(locs.state.unique())
    cat_list = cats.sort_values("orders", ascending=False).index.tolist()

    seller_view = sellers.sort_values("orders", ascending=False).copy()
    seller_view["label"] = (seller_view.seller_state + " · " + seller_view.seller_city.str.title() + " · "
                            + seller_view.orders.astype(int).astype(str) + " orders · late "
                            + (seller_view.late_rate * 100).round(0).astype(int).astype(str) + "% · "
                            + seller_view.seller_id.str[:6])
    NEW = "➕ New seller (no history)"
    seller_labels = [NEW] + seller_view.label.tolist()

    with st.form("order_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Product**")
            category = st.selectbox("Product category", cat_list, index=0)
            c = cats.loc[category]
            price = st.number_input("Price (R$)", min_value=1.0, value=float(round(c.median_price, 2)), step=5.0)
            freight = st.number_input("Freight (R$)", min_value=0.0, value=float(round(c.median_freight, 2)), step=1.0)
            weight = st.number_input("Weight (g)", min_value=1.0, value=float(max(c.median_weight_g, 1)), step=50.0)
            items = st.number_input("Items in order", min_value=1, value=1, step=1)
        with c2:
            st.markdown("**Seller**")
            seller_choice = st.selectbox("Seller", seller_labels)
            new_state = st.selectbox("(New seller) state", all_states, index=all_states.index("SP") if "SP" in all_states else 0)
            new_city = st.selectbox("(New seller) city", sorted(locs[locs.state == new_state].city.unique()))
            st.markdown("**Payment**")
            payment = st.selectbox("Payment type", [p for p in meta["payment_types"] if p != "unknown"])
            installments = st.number_input("Installments", min_value=1, value=1, step=1)
        with c3:
            st.markdown("**Customer & timing**")
            cust_state = st.selectbox("Customer state", all_states, index=all_states.index("SP") if "SP" in all_states else 0)
            cust_city = st.selectbox("Customer city", sorted(locs[locs.state == cust_state].city.unique()))
            purchase_dt = st.date_input("Purchase date", value=pd.Timestamp(meta["as_of"]).date())
            purchase_hour = st.slider("Purchase hour", 0, 23, 14)
            promised_override = st.number_input("Promised delivery window (days, 0 = route typical)", min_value=0.0, value=0.0, step=1.0)
        submitted = st.form_submit_button("Predict late-delivery risk", type="primary")

    if submitted:
        # seller info
        if seller_choice == NEW:
            seller_state, s_lat, s_lng = new_state, *city_coords(new_state, new_city)
            s_n, s_rate, s_rate90, s_orders30 = 0, G, G, 0
            seller_name = f"New seller ({new_city.title()}, {new_state})"
        else:
            s = seller_view[seller_view.label == seller_choice].iloc[0]
            seller_state, s_lat, s_lng = s.seller_state, s.seller_lat, s.seller_lng
            s_n, s_rate, s_rate90, s_orders30 = s.orders, s.hist_late_rate, s.seller_late_rate_90d, s.seller_orders_30d
            seller_name = f"Seller {s.seller_id[:8]}… ({s.seller_city.title()}, {s.seller_state})"

        c_lat, c_lng = city_coords(cust_state, cust_city)
        distance = haversine_km(s_lat, s_lng, c_lat, c_lng) if not np.isnan(s_lat) else np.nan

        route = f"{seller_state}->{cust_state}"
        if route in routes.index:
            r = routes.loc[route]
            route_rate, route_rate30, route_days, route_promised = r.hist_late_rate, r.route_late_rate_30d, r.route_hist_mean_days, r.median_promised_days
        else:
            route_rate, route_rate30, route_days, route_promised = G, G, meta["global_mean_delivery_days"], 24.0
        promised = promised_override if promised_override > 0 else float(route_promised)

        ts = pd.Timestamp(purchase_dt)
        row = pd.DataFrame([{
            "main_category": category, "customer_state": cust_state, "seller_state": seller_state, "payment_type": payment,
            "item_count": items, "unique_sellers": 1, "total_price": price, "total_freight": freight,
            "freight_ratio": freight / price, "total_weight_g": weight, "volume_cm3": c.median_volume_cm3,
            "distance_km": distance, "same_state": int(seller_state == cust_state), "promised_days": promised,
            "purchase_month": ts.month, "purchase_dow": ts.dayofweek, "purchase_hour": purchase_hour,
            "payment_installments": installments,
            "seller_hist_n": s_n, "seller_hist_late_rate": s_rate,
            "route_hist_late_rate": route_rate, "category_hist_late_rate": c.hist_late_rate,
            "platform_late_rate_14d": meta["platform_late_rate_14d"], "platform_orders_7d": meta["platform_orders_7d"],
            "seller_orders_30d": s_orders30, "seller_late_rate_90d": s_rate90,
            "route_late_rate_30d": route_rate30, "route_hist_mean_days": route_days,
            "buffer_vs_route": promised - route_days,
        }])[meta["features"]]

        p = float(model.predict_proba(row)[:, 1][0])
        band, icon = band_of(p)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Late-delivery risk", f"{p:.1%}", f"{p / G:.1f}× marketplace average ({G:.1%})", delta_color="inverse")
        m2.metric("Risk band", f"{icon} {band}")
        m3.metric("Delivery buffer", f"{promised - route_days:+.1f} days",
                  help="Promised window minus the route's historical average real delivery time. Negative = promise is tighter than reality.")
        st.progress(min(p / 0.5, 1.0))

        if band == "High":
            st.error("**Action:** high risk — prioritise with the carrier, consider a longer estimate, or send the customer a proactive update.")
        elif band == "Medium":
            st.warning("**Action:** moderate risk — keep on the watch-list.")
        else:
            st.success("**Action:** low risk — no intervention needed.")

        st.markdown("**What the model looked at**")
        d1, d2 = st.columns(2)
        d1.write(pd.DataFrame({
            "Signal": ["Seller", "Seller historical late rate", "Route", "Route historical late rate",
                       "Route average real delivery time", "Promised window", "Distance"],
            "Value": [seller_name, f"{s_rate:.1%} ({int(s_n)} past orders)", route, f"{route_rate:.1%}",
                      f"{route_days:.1f} days", f"{promised:.1f} days",
                      f"{distance:,.0f} km" if not np.isnan(distance) else "unknown"]}).set_index("Signal"))
        d2.write(pd.DataFrame({
            "Signal": ["Category late rate", "Marketplace late rate (last 14 days)", "Orders in the last 7 days"],
            "Value": [f"{c.hist_late_rate:.1%}", f"{meta['platform_late_rate_14d']:.1%}", f"{meta['platform_orders_7d']:,}"]}).set_index("Signal"))

# ----------------------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Seller scorecard")
    st.caption("Late rate is shrunk toward the marketplace average so sellers with few orders are not ranked unfairly. "
               "A seller needs 20+ orders to be classified.")

    k = st.columns(4)
    k[0].metric("Sellers", f"{len(sellers):,}")
    k[1].metric("Marketplace late rate", f"{G:.1%}")
    k[2].metric("High-risk sellers", int((sellers.status == "High risk").sum()))
    k[3].metric("Watch-list sellers", int((sellers.status == "Watch").sum()))

    f1, f2, f3 = st.columns(3)
    states = f1.multiselect("Seller state", sorted(sellers.seller_state.unique()))
    min_orders = f2.slider("Minimum orders", 1, 300, 30)
    statuses = f3.multiselect("Status", ["High risk", "Watch", "OK", "Not enough data"], default=["High risk", "Watch", "OK"])

    v = sellers[(sellers.orders >= min_orders) & (sellers.status.isin(statuses))]
    if states:
        v = v[v.seller_state.isin(states)]
    v = v.sort_values("smoothed_late_rate", ascending=False)

    show = v[["seller_id", "seller_state", "seller_city", "orders", "late_rate", "smoothed_late_rate",
              "avg_days_late_when_late", "avg_review_score", "status"]].copy()
    show["seller_city"] = show.seller_city.str.title()
    st.dataframe(
        show, hide_index=True, width="stretch",
        column_config={
            "late_rate": st.column_config.NumberColumn("Late rate", format="percent"),
            "smoothed_late_rate": st.column_config.NumberColumn("Smoothed late rate", format="percent"),
            "avg_days_late_when_late": st.column_config.NumberColumn("Avg days late (when late)", format="%.1f"),
            "avg_review_score": st.column_config.NumberColumn("Avg review", format="%.2f"),
            "orders": "Orders", "seller_id": "Seller", "seller_state": "State", "seller_city": "City", "status": "Status",
        })
    st.caption(f"{len(show):,} sellers shown")

    st.markdown("**Seller detail**")
    if len(show):
        pick = st.selectbox("Choose a seller", show.seller_id.tolist(), format_func=lambda x: x[:10] + "…")
        s = sellers[sellers.seller_id == pick].iloc[0]
        a = st.columns(5)
        a[0].metric("Orders", int(s.orders))
        a[1].metric("Late rate", f"{s.late_rate:.1%}", f"{(s.late_rate - G) * 100:+.1f} pp vs market", delta_color="inverse")
        a[2].metric("Late rate (last 90 days)", f"{s.seller_late_rate_90d:.1%}")
        a[3].metric("Avg review", f"{s.avg_review_score:.2f}" if pd.notna(s.avg_review_score) else "n/a")
        a[4].metric("Status", s.status)
        pct = (sellers[sellers.orders >= 30].late_rate < s.late_rate).mean()
        st.caption(f"Later than {pct:.0%} of sellers with 30+ orders.")

# ----------------------------------------------------------------------------- tab 3
with tab3:
    sat = meta["satisfaction"]
    st.subheader("How much does late delivery hurt satisfaction?")
    a = st.columns(4)
    a[0].metric("1–2 star reviews — on time", f"{sat['bad_review_rate_on_time']:.1%}")
    a[1].metric("1–2 star reviews — late", f"{sat['bad_review_rate_late']:.1%}", f"{sat['late_vs_on_time_ratio']:.1f}×", delta_color="inverse")
    a[2].metric("Avg score — on time / late", f"{sat['avg_score_on_time']:.2f} / {sat['avg_score_late']:.2f}")
    a[3].metric("Model-estimated extra bad-review risk", f"+{sat['counterfactual_extra_bad_review_pp']:.0f} pp")
    st.markdown("**1–2 star review rate by how late the order was**")
    st.bar_chart(buckets.set_index("delay_bucket")["bad_review_rate"])
    st.markdown("**Average review score by delay**")
    st.bar_chart(buckets.set_index("delay_bucket")["avg_score"])
    st.caption("The extra-risk figure is a model-based estimate (correlational, not a causal experiment). "
               "Satisfaction collapses once an order is more than ~3 days late.")

# ----------------------------------------------------------------------------- tab 4
with tab4:
    t = meta["test_metrics"]
    st.subheader("About the model")
    st.write(f"**{meta['model_name']}** trained on {meta['train_orders']:,} delivered orders "
             f"(9 joined tables). Evaluated on the most recent 20% of orders (out-of-time).")
    a = st.columns(5)
    a[0].metric("ROC-AUC", f"{t['roc_auc']:.2f}")
    a[1].metric("PR-AUC", f"{t['pr_auc']:.2f}", f"base rate {t['test_late_rate']:.1%}", delta_color="off")
    a[2].metric("Recall @ threshold", f"{t['recall']:.0%}")
    a[3].metric("Precision @ threshold", f"{t['precision']:.0%}")
    a[4].metric("Orders flagged", f"{t['flagged_share']:.0%}")
    st.markdown("**Observed late rate by risk band (test period)**")
    bs = pd.DataFrame(meta["band_summary"]).set_index("band")[["orders", "late_rate", "share_of_orders"]]
    st.dataframe(bs.style.format({"late_rate": "{:.1%}", "share_of_orders": "{:.0%}", "orders": "{:,.0f}"}))
    st.markdown("**Top drivers (permutation importance)**")
    st.bar_chart(imp.set_index("feature")["importance"].head(10))
    st.info("Lateness is only partly predictable at purchase time (strikes, weather and carrier incidents are invisible). "
            "Use this as a **risk-ranking tool** to prioritise attention, not as a guarantee.")
