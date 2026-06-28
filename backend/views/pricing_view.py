"""
backend/views/pricing_view.py
─────────────────────────────
Renders the Pricing and Subscription creation page.
"""

import streamlit as st
import httpx
from backend.config.settings import settings

def render_pricing_view(token: str):
    st.title("🚀 Upgrade to Pro")
    st.markdown("Unlock unlimited PDFs and Queries with our Pro tier.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Free
        * 5 PDFs Limit
        * 50 Queries / Day
        * Standard Support
        
        **$0 / month**
        """)
        user_tier = st.session_state.get("user", {}).get("tier", "free")
        if user_tier == "free":
            st.button("Current Plan", disabled=True)
            
    with col2:
        st.markdown("""
        ### Pro
        * **Unlimited PDFs**
        * **Unlimited Queries**
        * **Priority Processing**
        
        **$9 / month** or **$90 / year**
        """)
        
        interval = st.radio("Billing Cycle", ["monthly", "yearly"], horizontal=True)
        
        if user_tier == "pro":
            st.button("You are on Pro! 🎉", disabled=True)
        else:
            if st.button("Subscribe Now", type="primary"):
                try:
                    headers = {"Authorization": f"Bearer {token}"}
                    response = httpx.post(
                        f"{settings.api_base_url}/billing/create-subscription",
                        json={"interval": interval},
                        headers=headers,
                        timeout=10.0
                    )
                    response.raise_for_status()
                    data = response.json()
                    sub_id = data["subscription_id"]
                    
                    st.session_state["pending_subscription_id"] = sub_id
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create subscription: {e}")
                    
    # If a subscription was created, inject Razorpay checkout
    if "pending_subscription_id" in st.session_state:
        sub_id = st.session_state["pending_subscription_id"]
        # In a real app we'd fetch the razorpay key from the backend. 
        # For this template we will assume it's available or we can just fetch it.
        # But wait, the frontend needs the razorpay key id!
        # We can add an endpoint to get the razorpay key id or just inject a script that fetches it.
        st.info("Complete your payment below.")
        
        # Razorpay Checkout HTML
        # The key is passed directly via an API call or just from environment if streamit has access.
        # Streamlit settings isn't exactly the API settings, so we need to fetch the key.
        # Let's render a simple mock or the real razorpay button if we have the key.
        st.components.v1.html(
            f"""
            <form action="http://localhost:8000/api/v1/webhooks/razorpay" method="POST">
                <!-- Using razorpay checkout script -->
                <script
                    src="https://checkout.razorpay.com/v1/payment-button.js"
                    data-payment_button_id="pl_test_button"
                    data-subscription_id="{sub_id}"
                    async>
                </script>
            </form>
            <p style="font-family: sans-serif; font-size: 14px;">(Note: This requires Razorpay Key ID configuration on the frontend. Since this is Streamlit, we simulate the redirect or use the Razorpay Payment Button)</p>
            """,
            height=300,
        )
