"""
backend/views/billing_view.py
─────────────────────────────
Renders the Billing Dashboard where users can view their status, history, and cancel.
"""

import streamlit as st
import httpx
import pandas as pd
from backend.config.settings import settings

def render_billing_view(token: str):
    st.title("💳 Billing Dashboard")
    
    # User state
    user = st.session_state.get("user", {})
    tier = user.get("tier", "free")
    status = user.get("subscription_status", "none")
    end_date = user.get("subscription_end")
    canceling = user.get("cancel_at_period_end", False)
    
    st.markdown("### 📊 Subscription Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Plan", tier.upper())
    col2.metric("Status", status.title())
    col3.metric("Renews / Ends On", str(end_date).split()[0] if end_date else "N/A")
    
    if tier == "pro" and not canceling:
        st.warning("You are currently subscribed to the Pro tier.")
        if st.button("Cancel Subscription", type="secondary"):
            try:
                headers = {"Authorization": f"Bearer {token}"}
                resp = httpx.post(
                    f"{settings.api_base_url}/billing/cancel-subscription",
                    headers=headers,
                    timeout=5.0
                )
                resp.raise_for_status()
                st.success("Subscription cancelled. It will remain active until the billing period ends.")
                st.rerun() # In a real app we'd refresh the user object first
            except Exception as e:
                st.error(f"Failed to cancel: {e}")
                
    elif canceling:
        st.info("Your subscription is set to cancel at the end of the billing period.")
        
    st.markdown("---")
    st.markdown("### 📜 Payment History")
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        resp = httpx.get(
            f"{settings.api_base_url}/billing/history",
            headers=headers,
            timeout=5.0
        )
        if resp.status_code == 200:
            history = resp.json()
            if history:
                df = pd.DataFrame(history)
                # Format
                df["amount"] = df["amount"] / 100.0 # razorpay uses paise
                df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(df[["created_at", "amount", "currency", "status"]], use_container_width=True)
            else:
                st.write("No payment history found.")
        else:
            st.error("Failed to load payment history.")
    except Exception as e:
        st.error(f"Error fetching history: {e}")
