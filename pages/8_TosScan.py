import streamlit as st
import imaplib
import email
import re
import datetime
import pandas as pd
from dateutil import parser
import time
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_session_state():
    """Initialize session state variables"""
    if 'processed_email_ids' not in st.session_state:
        st.session_state.processed_email_ids = set()
    if 'cached_data' not in st.session_state:
        st.session_state.cached_data = {}
    if 'previous_symbols' not in st.session_state:
        st.session_state.previous_symbols = {}

# Fetch credentials from Streamlit Secrets
EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

# Constants
SENDER_EMAIL = "alerts@thinkorswim.com"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Define keywords for scan types
KEYWORDS = ["HG_30mins_L", "HG_30mins_S"]

def connect_to_email(retries=MAX_RETRIES):
    """Establish email connection with retry logic."""
    for attempt in range(retries):
        try:
            mail = imaplib.IMAP4_SSL('imap.gmail.com')
            mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            return mail
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            time.sleep(RETRY_DELAY)

def parse_email_body(msg):
    """Parse email body with better HTML handling."""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() in ["text/plain", "text/html"]:
                    body = part.get_payload(decode=True).decode()
                    if part.get_content_type() == "text/html":
                        soup = BeautifulSoup(body, "html.parser", from_encoding='utf-8')
                        return soup.get_text(separator=' ', strip=True)
                    return body
        else:
            body = msg.get_payload(decode=True).decode()
            if msg.get_content_type() == "text/html":
                soup = BeautifulSoup(body, "html.parser", from_encoding='utf-8')
                return soup.get_text(separator=' ', strip=True)
            return body
    except Exception as e:
        logger.error(f"Error parsing email body: {e}")
        return ""

def extract_stock_symbols_from_email(email_address, password, sender_email, keyword, days_lookback):
    """Extract stock symbols from email alerts with proper date filtering."""
    if keyword in st.session_state.cached_data:
        return st.session_state.cached_data[keyword]

    try:
        mail = connect_to_email()
        mail.select('inbox')

        today = datetime.date.today()
        start_date = today
        if days_lookback > 1:
            start_date = today - datetime.timedelta(days=days_lookback-1)
        
        date_since = start_date.strftime("%d-%b-%Y")
        search_criteria = f'(FROM "{sender_email}" SUBJECT "{keyword}" SINCE "{date_since}")'
        _, data = mail.search(None, search_criteria)

        stock_data = []
        
        for num in data[0].split():
            if num in st.session_state.processed_email_ids:
                continue

            _, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            
            email_datetime = parser.parse(msg['Date'])
            email_date = email_datetime.date()
            
            if email_date < start_date or email_datetime.weekday() >= 5:
                continue

            body = parse_email_body(msg)
            symbols = re.findall(r'New symbols:\s*([A-Z,\s]+)\s*were added to\s*(' + re.escape(keyword) + ')', body)
            
            if symbols:
                for symbol_group in symbols:
                    extracted_symbols = symbol_group[0].replace(" ", "").split(",")
                    signal_type = symbol_group[1]
                    for symbol in extracted_symbols:
                        if symbol.isalpha():
                            stock_data.append([symbol, email_datetime, signal_type])
            
            st.session_state.processed_email_ids.add(num)

        mail.close()
        mail.logout()

        if stock_data:
            df = pd.DataFrame(stock_data, columns=['Ticker', 'Date', 'Signal'])
            df = df.sort_values(by=['Date', 'Ticker']).drop_duplicates(subset=['Ticker', 'Signal', 'Date'], keep='last')
            st.session_state.cached_data[keyword] = df
            return df

        empty_df = pd.DataFrame(columns=['Ticker', 'Date', 'Signal'])
        st.session_state.cached_data[keyword] = empty_df
        return empty_df

    except Exception as e:
        logger.error(f"Error in extract_stock_symbols_from_email: {e}")
        st.error(f"Error processing emails: {str(e)}")
        return pd.DataFrame(columns=['Ticker', 'Date', 'Signal'])

def get_new_symbols_count(keyword, current_df):
    """Get count of new symbols."""
    if current_df.empty:
        return 0

    current_symbols = set(current_df['Ticker'].unique())
    previous_symbols = st.session_state.previous_symbols.get(keyword, set())
    
    new_symbols = current_symbols - previous_symbols
    st.session_state.previous_symbols[keyword] = current_symbols
    
    return len(new_symbols)

def run():
    """Main function to run the Streamlit application"""
    init_session_state()
    
    st.title("📊 TOS Scan Alerts")
    
    # Simple settings in sidebar
    with st.sidebar:
        st.header("Settings")
        days_lookback = st.slider(
            "Days to Look Back",
            min_value=1,
            max_value=3,
            value=2
        )
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.session_state.cached_data.clear()
        st.session_state.processed_email_ids.clear()
        st.rerun()

    # Display scans
    for keyword in KEYWORDS:
        symbols_df = extract_stock_symbols_from_email(
            EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback
        )
        
        new_count = get_new_symbols_count(keyword, symbols_df)
        
        header = f"📊 {keyword}"
        if new_count > 0:
            header = f"📊 {keyword} 🔴 {new_count} new"
        
        with st.expander(header, expanded=True):
            if not symbols_df.empty:
                display_df = symbols_df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(display_df, use_container_width=True)
                
                csv = symbols_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download {keyword}",
                    data=csv,
                    file_name=f"{keyword}_{datetime.date.today()}.csv",
                    mime="text/csv",
                )
            else:
                st.warning(f"No signals found for {keyword} in the last {days_lookback} day(s).")

if __name__ == "__main__":
    run()