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
import yfinance as yf
from collections import defaultdict
import pytz

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

@st.cache_data(ttl=3600)
def get_ticker_info(ticker):
    """Fetch ticker info from yfinance with caching."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'name': info.get('longName', ticker)
        }
    except Exception as e:
        logger.error(f"Error fetching info for {ticker}: {e}")
        return {'sector': 'Unknown', 'industry': 'Unknown', 'name': ticker}

def generate_summary(all_data):
    """Generate industry-grouped summary of all tickers."""
    if all_data.empty:
        return None
    
    # Get unique tickers
    unique_tickers = all_data['Ticker'].unique()
    
    # Fetch info for all tickers
    ticker_info_map = {}
    progress_bar = st.progress(0, text="Fetching industry data...")
    for i, ticker in enumerate(unique_tickers):
        ticker_info_map[ticker] = get_ticker_info(ticker)
        progress_bar.progress((i + 1) / len(unique_tickers), text=f"Fetching data for {ticker}...")
    progress_bar.empty()
    
    # Group by industry
    industry_groups = defaultdict(lambda: {'long': [], 'short': []})
    
    for _, row in all_data.iterrows():
        ticker = row['Ticker']
        signal = row['Signal']
        info = ticker_info_map[ticker]
        industry = info['industry']
        
        if 'HG_30mins_L' in signal:
            industry_groups[industry]['long'].append(ticker)
        else:
            industry_groups[industry]['short'].append(ticker)
    
    return industry_groups, ticker_info_map

def run():
    """Main function to run the Streamlit application"""
    init_session_state()
    
    st.title("📊 TOS Scan Alerts")
    
    # Get current date in US Eastern Time (market timezone)
    eastern = pytz.timezone('US/Eastern')
    today_et = datetime.datetime.now(eastern).date()
    
    # Simple settings in sidebar
    with st.sidebar:
        st.header("Settings")
        selected_date = st.date_input(
            "Select Date (ET)",
            value=today_et,
            max_value=today_et,
            min_value=today_et - datetime.timedelta(days=7)
        )
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.session_state.cached_data.clear()
        st.session_state.processed_email_ids.clear()
        st.rerun()

    # Collect all data first
    all_dataframes = []
    
    # Display scans
    for keyword in KEYWORDS:
        # Fetch last 7 days of data
        symbols_df = extract_stock_symbols_from_email(
            EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, 7
        )
        
        # Filter to selected date only
        if not symbols_df.empty:
            symbols_df['DateOnly'] = pd.to_datetime(symbols_df['Date']).dt.date
            symbols_df = symbols_df[symbols_df['DateOnly'] == selected_date].copy()
            symbols_df = symbols_df.drop('DateOnly', axis=1)
        
        if not symbols_df.empty:
            all_dataframes.append(symbols_df)
        
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
                st.warning(f"No signals found for {keyword} on {selected_date.strftime('%Y-%m-%d')}.")
    
    # Generate summary section
    if all_dataframes:
        st.markdown("---")
        st.subheader("📋 Trader Summary")
        st.caption(f"Signals from {selected_date.strftime('%B %d, %Y')}")
        
        all_data = pd.concat(all_dataframes, ignore_index=True)
        
        # Get latest signal per ticker (in case of multiple on same date)
        all_data = all_data.sort_values('Date').groupby('Ticker').last().reset_index()
        
        summary_result = generate_summary(all_data)
        
        if summary_result:
            industry_groups, ticker_info_map = summary_result
            
            # Create shareable text summary
            summary_text = f"**TOS High Grade 30-Min Signals Summary - {selected_date.strftime('%B %d, %Y')}**\n\n"
            summary_text += f"_Date: {selected_date.strftime('%B %d, %Y')} | Interval: 30 minutes_\n\n"
            
            for industry in sorted(industry_groups.keys()):
                signals = industry_groups[industry]
                if signals['long'] or signals['short']:
                    summary_text += f"**{industry}**\n"
                    if signals['long']:
                        summary_text += f"  • LONG: {', '.join(sorted(set(signals['long'])))}\n"
                    if signals['short']:
                        summary_text += f"  • SHORT: {', '.join(sorted(set(signals['short'])))}\n"
                    summary_text += "\n"
            
            # Display in expandable section
            with st.expander("📊 Industry-Grouped Summary (Click to expand)", expanded=True):
                st.markdown(summary_text)
                
                # Copy to clipboard button
                st.code(summary_text, language=None)
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tickers", len(all_data))
                with col2:
                    long_count = len(all_data[all_data['Signal'].str.contains('_L')])
                    st.metric("Long Signals", long_count)
                with col3:
                    short_count = len(all_data[all_data['Signal'].str.contains('_S')])
                    st.metric("Short Signals", short_count)

if __name__ == "__main__":
    run()