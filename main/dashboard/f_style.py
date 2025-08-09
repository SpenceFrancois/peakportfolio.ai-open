import streamlit as st

def apply_custom_css():
    """
    Applies modular custom CSS to the Streamlit app for improved styling,
    with all responsive adjustments centralized into one block.
    The title header sizes are now consistent on larger screens but become
    a little smaller on mobile devices. Additionally, on mobile devices the
    table font size is reduced and adjustments are applied to prevent long words
    from touching the cell edges.
    """
    

    
    # Base CSS for components without media queries
    base_css = """
        /* Global Font Override using universal selector */

        }

        /* Center Sidebar Contents */
        [data-testid="stSidebar"] {
            text-align: center;
        }


        /* Global Wrapper to constrain max width */
        .main-wrapper {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Center Container */
        .center-container {
            text-align: center;
            margin-top: 0px;
            margin-left: auto;
            margin-right: auto;
            max-width: 1200px;
        }

        /* Title and Subheader Styling */
        h1, h2 {
            color: #0a4daa;
            text-align: left;
            font-size: 22px; /* Fixed size for consistency on larger screens */
            padding-bottom: 5px;
            margin-bottom: 20px;
            margin-top: 60px; /* Increased top margin for better spacing */
            border-bottom: 2px solid #0a4daa;
        }

        

        /* Target the dropdown container and limit its height  specifically for the start and end dates*/
        ul[data-testid="stSelectboxVirtualDropdown"] > div {
            height: 150px !important;  /* Adjust this value as needed (150px ≈ 5 rows) */
            max-height: 150px !important;
            overflow-y: auto !important;
        }

        




    /* Parent becomes a simple flex container */
    [data-testid="stButton"] {
        display: flex;
    }
    [data-testid="stButton"] button,
    [data-testid="stButton"] button * {
    font-size: 18px !important;
    font-weight: 600 !important;
    }
        
    /* Give the button permission to grow and fill all available space */
    [data-testid="stButton"] button {
        flex: 1;
        position: relative;
        overflow: hidden;
        background: linear-gradient(90deg, #0a4daa 0%, #007bb8 100%);
        color: #fff !important;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        cursor: pointer;
        text-align: center;
    }

    /* Light‐sweep pseudo‐element */
    [data-testid="stButton"] button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -75%;
        width: 50%;
        height: 100%;
        background: rgba(255,255,255,0.3);
        transform: skewX(-25deg);
        transition: transform 0.7s ease;
    }

    /* trigger sweep on focus and hide default outline */
    [data-testid="stButton"] button:focus::before {
        transform: translateX(380%) skewX(-25deg);
        outline: none;
    }

    /* Remove press feedback */
    [data-testid="stButton"] button:active,
    [data-testid="stButton"] button:hover {
        transform: none;
    }


    /* subtle press feedback */
    [data-testid="stButton"] button:active {
        transform: scale(.98);
    }

    /* Mobile tweaks */
    @media (max-width: 480px) {
        [data-testid="stButton"] button {
            padding: 12px 24px;
            font-size: 20px;
        }
    }







    

    .stTextArea {
        border: 1px solid #0a4daa !important;
        border-radius: 8px;
        padding-left: 10px;
        padding-right: 10px;
        padding-bottom: 15px;
    }

        details {
            background-color: #fdfeff !important;
        }

        /* Input Field Styling */
        .stTextInput, .stNumberInput, .stDateInput, .stSelectbox, .stSlider, .stCheckbox {
            border: 1px solid #0a4daa !important;
            border-radius: 4px;
            padding: 5px !important;
            padding-bottom: 15px !important;
        }

        details {
            border: 1px solid #0a4daa !important;
        }
        
        .stAlertContainer {
            background-color: #e8f5ff !important; /* Darker Blue */
        }
        .stAlertContainer p, 
        .stAlertContainer div {
            color: #0a4daa !important; /* Dark Blue */
        }

        /* AI-Specific Styling */
        .stChatInput textarea {
            border: 1px solid #0a4daa !important;
            border-radius: 8px;
            box-sizing: border-box;
        }
        .custom-text-box {
            border: 1px solid #ccc;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 15px;
            border: 2px solid #0a4daa !important;
            background-color: #FFFFFF;
            max-height: 700px;
            overflow-y: auto;
            max-width: 1200px;
        }



        .custom-input:focus 
            box-shadow: 0 0 4px rgba(0, 123, 255, 0.5);
        }

        .custom-chat-message {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
            max-width: 1200px;
            margin-top: 10px;
        }

        /* Risk Reward Portfolio Table */
        [data-testid="stDataFrameResizable"] {
            border: 1px solid #0a4daa !important;
        }

        















        /* Correlation Table CSS */
        .correlation-table-container {
            width: 100%;
            max-width: 1200px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
            background: #fff;
            border: 1px solid #0a4daa;
            margin-bottom: 10px;
        }

        .correlation-table {
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            font: 1em sans-serif;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
            /* Remove overflow: hidden to prevent clipping sticky elements */
            table-layout: auto;
        }


        .correlation-table tbody tr:nth-of-type(even) {
            background: #f9f9f9;
        }

        .correlation-table thead th {
            background: #427ecd;
        }

        .correlation-table tbody td {
            background: #fff;
            color: #000;
        }

        .correlation-table tbody tr:last-of-type td {
            border-bottom: none;
        }

        /* Sticky first column */
        .correlation-table th:first-child,
        .correlation-table td:first-child {
            position: -webkit-sticky;
            position: sticky;
            left: 0;
            background: #fff;
            z-index: 2;
        }

        /* Ensure header cell in first column stays on top */
        .correlation-table thead th:first-child {
            z-index: 3;
        }


        
        
















        /* Styled Frontier CSS */
        .styled-frontier {
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            font: 1.1em sans-serif;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
            overflow: hidden;
            table-layout: auto;
            margin: 0 auto;
            border: 1px solid #0a4daa;
        }


        










        /* Container for centering and horizontal scroll */
        .styled-table-container {
            width: 100%;
            max-width: 1200px;
            margin: 40px auto;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0;
        }

        /* Main table styling: overall border and rounded corners */
        .styled-table {
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            min-width: 350px;
            font: 1em sans-serif;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border: 1px solid #0a4daa;
            border-radius: 10px;
            overflow: hidden;
            table-layout: fixed;
            margin: auto;
        }

        /* Ensure the table header remains aligned */
        .styled-table thead {
            display: table;
            width: 100%;
            table-layout: fixed;
        }

        /* Header row styling */
        .styled-table thead tr {
            background: #0a4daa;
            color: #ffffff;
            text-align: center;
            font-weight: bold;
        }

        /* Header cells and body cells styling */
        .styled-table th,
        .styled-table td {
            text-align: center;
            vertical-align: middle;
            padding: 10px 0;
            word-wrap: break-word;
            white-space: normal;
            overflow: hidden;
        }

        /* Apply scroll behavior to the tbody for tables without the .non-scrollable class */
        .styled-table:not(.non-scrollable) tbody {
            display: block;
            max-height: 600px; /* Adjust as needed */
            overflow-y: auto;
        }

        /* Maintain column alignment between thead and tbody rows */
        .styled-table:not(.non-scrollable) thead,
        .styled-table:not(.non-scrollable) tbody tr {
            display: table;
            width: 100%;
            table-layout: fixed;
        }

        /* Body row styling */
        .styled-table tbody tr {
            background: #fff;
            color: #000;
            border-collapse: collapse;
        }

        /* Alternate row background for better readability */
        .styled-table tbody tr:nth-of-type(even) {
            background: #f9f9f9;
            border-collapse: collapse;
        }

        /* Remove bottom border from the last row */
        .styled-table tbody tr:last-of-type td {
            border-bottom: none;
            border-collapse: collapse;
        }

        /* Hover effect for entire row */
        .styled-table tbody tr:hover {
            background-color: #e6f2ff;
        }


        

    .custom-table-wrapper {
        max-width: 100%;
        max-height: 1000px;
        overflow-x: auto;
        border-radius: 10px;
        border: 1px solid #0a4daa; /* outer border only */
        background-color: white; /* ensures background doesn't leak through */
    }

    /* Table itself without outer borders — inner grid only */
    .custom-portfolio-table {
        min-width: 900px;
        width: max-content;
        border-collapse: separate;
        border-spacing: 0;
        font-family: 'Arial', sans-serif;
        border: none;
        border-radius: 10px;
        table-layout: fixed;
    }

    /* Sticky header */
    .custom-portfolio-table thead th {
        position: sticky;
        top: 0;
        background-color: #0a4daa;
        color: white;
        z-index: 2;
        padding: 8px 10px;
        text-align: center;
        white-space: nowrap;
    }

    /* Sticky first column (header + body) */
    .custom-portfolio-table thead th:first-child,
    .custom-portfolio-table td:first-child {
        position: sticky;
        left: 0;
        z-index: 2;
        min-width: 60px;
        text-align: center;
        white-space: nowrap;
    }

    /* Leftmost header cell */
    .custom-portfolio-table thead th:first-child {
        background-color: #0a4daa;
        color: white;
        z-index: 3;
    }

    /* Leftmost body cells — match row backgrounds */
    .custom-portfolio-table td:first-child {
        background-color: inherit;
        color: inherit;
    }

    /* Table body cells with inner simulated borders only */
    .custom-portfolio-table td {
        padding: 8px 10px;
        text-align: center;
        white-space: nowrap;
        color: #333;
        background-clip: padding-box;
        border-bottom: 1px solid #e0e0e0;
        border-right: 1px solid #e0e0e0;
    }

    /* Zebra striping */
    .custom-portfolio-table tbody tr:nth-child(even) {
        background-color: #f9f9f9;
    }

    .custom-portfolio-table tbody tr:nth-child(odd) {
        background-color: #ffffff;
    }

    /* Hover effect for entire row */
    .custom-portfolio-table tbody tr:hover {
        background-color: #e6f2ff;
    }

    /* Rounded bottom corners (no double borders) */
    .custom-portfolio-table tbody tr:last-child td:first-child {
        border-bottom-left-radius: 10px;
    }

    .custom-portfolio-table tbody tr:last-child td:last-child {
        border-bottom-right-radius: 10px;
    }

    /* Optional: remove right border on last cell to align cleanly with wrapper */
    .custom-portfolio-table tbody td:last-child,
    .custom-portfolio-table thead th:last-child {
        border-right: none;
    }






        




    """
    
    # Global Media Queries for Responsive Adjustments
    global_media_queries = """
        @media (max-width: 768px) {
            /* Adjust Headers on mobile */
            h1, h2 {
                font-size: 18px;
            }
            /* Adjust Buttons */
            .stButton button {
                width: 100%;
                font-size: 1rem;
                padding: 10px;
            }
            /* Adjust AI Components */
            .custom-text-box {
                max-height: 600px;
            }
            .custom-chat-message {
                padding: 12px;
                font-size: 0.9rem;
            }
            /* Adjust Styled Tables */
            .styled-table {
                font-size: 0.68em; /* Reduced from 0.85em */
                min-width: unset;
            }
            .styled-table th,
            .styled-table td {
                padding: 10px 4px;
                box-sizing: border-box;
                word-wrap: normal;
                overflow-wrap: normal;
            }
        }

        @media (max-width: 376px) {
            .styled-table {
                font-size: 0.76em; /* General table text size */
                min-width: unset;
            }

            .styled-table th {
                font-size: 0.95em; 
                padding: 8px 4px; /* Slightly more padding */
            }

            .styled-table td {
                padding: 6px 5px;
                font-size: inherit; /* Inherits from .styled-table */
                white-space: normal;
                overflow-x: auto;
            }

            .styled-table th,
            .styled-table td {
                box-sizing: border-box;
                word-wrap: normal;
                overflow-wrap: normal;
            }
        }

    """
    
    # Combine the base CSS with the global media queries
    combined_css = base_css + global_media_queries
    
    # Inject the CSS into the Streamlit app
    st.markdown(
        f"""
        <style>
            /* Constrain the block container globally */
            .block-container {{
                max-width: 1200px;
                padding: 0 20px;
            }}
            {combined_css}
        </style>
        """,
        unsafe_allow_html=True
    )

