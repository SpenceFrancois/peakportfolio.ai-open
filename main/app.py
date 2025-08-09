import streamlit as st
import firebase_admin
from firebase_admin import auth, credentials, firestore

# Configure the page
st.set_page_config(page_title="PeakPortfolio", page_icon="FAVI.png", layout="wide")

# Sidebar Shadow
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        box-shadow: 4px 0px 10px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Global styles and header text effect
st.markdown(
    """
    <style>
        /* Hide default header/footer */
        header, .st-emotion-cache-z5fcl4, footer { 
            display: none !important;
        }
        html, body, [data-testid="stAppViewContainer"] {
            margin: 0;
            padding: 0;
        }
        
        /* Navbar styling */
        .auth-btn-wrapper {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background: #fdfdfd;
            padding-right: 40px;
            padding-top: 12px;
            padding-bottom: 10px;
            z-index: 1000;
            display: flex;
            justify-content: flex-end;
            align-items: center;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
        }
        .auth-btn-container {
            display: flex;
            align-items: center;
            gap: 17px;
        }
        /* Apply common header text effect */
        a.auth-btn, .status-display {
            color: #0a4daa;  /* Default text color */
            font-size: 16px;
            font-weight: bold;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .auth-btn {
            padding: 0;
        }
        /* Hover effect for header text */
        .auth-btn:hover, .status-display:hover {
            background: #0a4daa;
            color: white;
            border-radius: 8px;
            text-decoration: none;
            padding-left: 4px;
            padding-right: 4px;
        }
        
        /* New Premium User styling */
        .premium-user {
            color: #b8860b; /* Dark gold */
            font-size: 16px;
            font-weight: bold;
            text-decoration: none;
            cursor: not-allowed;
            transition: all 0.3s ease;
        }

        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .auth-btn-wrapper {
                padding-right: 10px;
                padding-top: 10px;
            }
            .auth-btn-container {
                margin-top: 3px;
            }
            .auth-btn, .status-display {
                font-size: 15px;
            }
        }
        @media (max-width: 400px) {
            .auth-btn-container {
                flex-wrap: wrap;
                gap: 8px;
            }
            .auth-btn-wrapper {
                padding-right: 8px;
                padding-top: 8px;
            }
            .auth-btn {
                font-size: 15px;
                padding: 3px;
            }
            .status-display {
                font-size: 15px;
            }
        }
        
        /* Footer styling */
        .footer {
            position: fixed;
            bottom: 10px;
            right: 20px;
            background: white;
            font-size: 14px;
            font-weight: bold;
            padding: 5px 10px;
            z-index: 1000;
            text-align: right;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Adjust header button position
st.markdown(
    """
    <style>
    [data-testid="stBaseButton-headerNoPadding"] {
        position: relative;
        top: -15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Firebase Admin SDK Initialization
if not firebase_admin._apps:
    cred = credentials.Certificate("main/.secrets/peakportfolio-df731-firebase-adminsdk-fdfvq-ffbadb18e1.json")
    firebase_admin.initialize_app(cred)

def verify_firebase_token(token):
    """
    Verifies the Firebase ID token and returns the user's UID if valid.
    """
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token.get("uid")
    except Exception:
        return None

def get_display_name(user_uid):
    """
    Retrieves the user's display name or falls back to the UID.
    """
    user_record = auth.get_user(user_uid)
    return user_record.display_name if user_record.display_name else user_uid

def get_active_subscription(user_uid):
    """
    Queries Firestore for an active subscription under customers/{user_uid}/subscriptions.
    Returns the subscription data if one is found, otherwise None.
    """
    try:
        db = firestore.client()
        subscriptions_ref = db.collection("customers").document(user_uid).collection("subscriptions")
        query = subscriptions_ref.where("status", "==", "active")
        docs = query.stream()
        for doc in docs:
            return doc.to_dict()
        return None
    except Exception as e:
        st.error("Error retrieving subscription data.")
        return None

def get_plan_name(product_id):
    """
    Maps a product ID to a human-readable plan name.
    """
    plan_mapping = {
        "prod_RdFK1Kahla0Dzj": "Premium",
        "prod_enterprise": "Enterprise",
        # Extend this dictionary for additional tiers as needed
    }
    return plan_mapping.get(product_id, "Subscribed")

def main():
    # Retrieve or initialize the user's UID from session state
    user_uid = st.session_state.get('user_uid', None)
    if not user_uid:
        query_params = st.experimental_get_query_params()
        token = query_params.get("token", [None])[0]
        if token:
            user_uid = verify_firebase_token(token)
            st.session_state.user_uid = user_uid
            # Remove the token from the URL
            st.experimental_set_query_params()
        else:
            st.session_state.user_uid = None

    # Build the status display (user's display name or a login prompt)
    if user_uid:
        display_name = get_display_name(user_uid)
        status_html = f'<div class="status-display">Logged-In</div>'
    else:
        status_html = (
            '<div class="status-display">'
            '<a href="https://www.peakportfolio.ai/user-pages/login" class="auth-btn">'
            'Login / Sign Up'
            '</a></div>'
        )

    # Check Firestore for an active subscription
    subscription = get_active_subscription(user_uid) if user_uid else None
    product_id = subscription.get("product", "") if subscription else None

    # Dynamically build the Premium button based on product_id existence
    if product_id:
        premium_button_html = (
            '<div class="premium-user" style="cursor: not-allowed;">'
            'Premium'
            '</div>'
        )
    else:
        premium_button_html = (
            '<a href="https://peakportfolio.ai/#Pricing" class="auth-btn">'
            'Unlock Pro'
            '</a>'
        )

    # Render the navbar with the Premium button, Home link, and status display
    st.markdown(
        f"""
        <div class="auth-btn-wrapper">
            <div class="auth-btn-container">
                <a href="https://peakportfolio.ai" class="auth-btn">Home</a>
                {premium_button_html}
                {status_html}
            </div>
        </div>
        <!-- Footer -->
        <div class="footer">
            PeakPortfolio™
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load the Dashboard Page, passing the active product (i.e., product_id)
    from dashboard.f_dashboard import dashboard_page
    from dashboard.f_style import apply_custom_css  
    apply_custom_css()
    dashboard_page(product_id=product_id)

    st.write('')

if __name__ == "__main__":
        main()

