"""
Shared UI Components for Options Trading Platform
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

def display_page_header(title: str, subtitle: str = "", icon: str = "📊"):
    """Display consistent page header across all pages"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                border-left: 4px solid #00ff88;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="font-size: 3rem;">{icon}</span>
            <div>
                <h1 style="margin: 0; color: #00ff88;">{title}</h1>
                <p style="margin: 0; color: #888; font-size: 1.1rem;">{subtitle}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_metric_card(label: str, value: str, delta: str = None, color: str = "#00ff88"):
    """Display metric card with optional delta"""
    delta_html = ""
    if delta:
        delta_color = "green" if "+" in delta else "red"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.9rem; margin-top: 0.5rem;">{delta}</div>'
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%);
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #333;">
        <div style="font-size: 0.9rem; color: #888; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 2rem; font-weight: bold; color: {color};">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def display_signal_badge(signal_type: str, strength: float = None):
    """Display signal type badge"""
    color_map = {
        'BUY': '#00ff88',
        'SELL': '#ff0088',
        'NEUTRAL': '#888',
        'BULLISH': '#00ff88',
        'BEARISH': '#ff0088'
    }
    
    color = color_map.get(signal_type.upper(), '#888')
    strength_text = f" ({strength:.0%})" if strength else ""
    
    st.markdown(f"""
    <span style="background: {color};
                 color: #000;
                 padding: 0.3rem 0.8rem;
                 border-radius: 20px;
                 font-weight: bold;
                 font-size: 0.9rem;">
        {signal_type.upper()}{strength_text}
    </span>
    """, unsafe_allow_html=True)


def display_status_indicator(status: str, label: str = "Status"):
    """Display system status indicator"""
    status_map = {
        'online': ('🟢', '#00ff88', 'Online'),
        'offline': ('🔴', '#ff0088', 'Offline'),
        'warning': ('🟡', '#ffaa00', 'Warning'),
        'loading': ('🔵', '#0088ff', 'Loading')
    }
    
    emoji, color, text = status_map.get(status.lower(), ('⚪', '#888', 'Unknown'))
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem;">
        <span style="font-size: 1.2rem;">{emoji}</span>
        <span style="color: {color}; font-weight: bold;">{label}: {text}</span>
    </div>
    """, unsafe_allow_html=True)


def create_data_table_with_filters(df, key_prefix="table"):
    """Create interactive data table with filtering"""
    if df.empty:
        st.info("No data available")
        return
    
    # Filters
    with st.expander("🔍 Filters"):
        cols = st.columns(3)
        filters = {}
        
        for idx, col in enumerate(df.columns[:3]):
            with cols[idx % 3]:
                if df[col].dtype in ['int64', 'float64']:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    filters[col] = st.slider(
                        f"{col}",
                        min_val,
                        max_val,
                        (min_val, max_val),
                        key=f"{key_prefix}_{col}"
                    )
    
    # Apply filters
    filtered_df = df.copy()
    for col, (min_val, max_val) in filters.items():
        if col in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df[col] >= min_val) & 
                (filtered_df[col] <= max_val)
            ]
    
    # Display
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Export
    if st.button("📥 Export to CSV", key=f"{key_prefix}_export"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key=f"{key_prefix}_download"
        )


def display_quick_nav():
    """Display quick navigation buttons"""
    st.markdown("---")
    st.markdown("### 🚀 Quick Navigation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("Main_Dashboard.py")
    
    with col2:
        if st.button("📈 Gamma Scanner", use_container_width=True):
            st.switch_page("pages/2_📈_Max_Gamma_Scanner.py")
    
    with col3:
        if st.button("🎯 Boundary Scanner", use_container_width=True):
            st.switch_page("pages/4_🎯_Boundary_Scanner.py")
    
    with col4:
        if st.button("💎 Opportunities", use_container_width=True):
            st.switch_page("pages/6_💎_Opportunity_Scanner.py")


def display_footer():
    """Display consistent footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Options Trading Platform | Built with Streamlit</p>
        <p><em>⚠️ For educational purposes only. Not financial advice.</em></p>
    </div>
    """, unsafe_allow_html=True)


def apply_custom_theme():
    """Apply custom CSS theme"""
    st.markdown("""
    <style>
        /* Global theme */
        :root {
            --primary-color: #00ff88;
            --secondary-color: #0088ff;
            --background-dark: #0e1117;
            --card-background: #1e1e1e;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e1e1e 0%, #0e1117 100%);
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 5px;
            border: 1px solid var(--primary-color);
            background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
            color: var(--primary-color);
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background: var(--primary-color);
            color: #000;
            transform: scale(1.05);
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            color: var(--primary-color);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #1e1e1e;
            border-radius: 5px;
            color: #888;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #00ff88 0%, #0088ff 100%);
            color: #000;
        }
    </style>
    """, unsafe_allow_html=True)
