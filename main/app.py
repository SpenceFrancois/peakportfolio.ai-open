import streamlit as st

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

def main():
    # Keep session keys for downstream compatibility (if any module expects them)
    st.session_state.user_uid = None

    # Open-source edition: no subscription/auth logic
    product_id = None

    # Render the navbar with Open-Source-Edition only
    st.markdown(
        """
        <div class="auth-btn-wrapper">
            <div class="auth-btn-container">
                <div class="status-display">Open-Source-Edition</div>
            </div>
        </div>
        <!-- Footer -->
        <div class="footer">
            PeakPortfolio™
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load the Dashboard Page, passing product_id for downstream compatibility
    from dashboard.f_dashboard import dashboard_page
    from dashboard.f_style import apply_custom_css
    apply_custom_css()
    dashboard_page(product_id=product_id)

    st.write('')

if __name__ == "__main__":
    main()

