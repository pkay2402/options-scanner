"""
AI Stock Report Generator
Generate comprehensive Substack-style investment articles using LLM

Requires: openai>=1.0.0, anthropic>=0.18.0
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logger = logging.getLogger(__name__)


def get_api_key_from_secrets(key_name: str) -> tuple[str | None, str]:
    """
    Get API key from Streamlit secrets or environment variables.
    
    Returns:
        Tuple of (api_key, source) where source is 'secrets', 'env', or 'none'
    """
    # Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name], 'secrets'
    except Exception:
        pass
    
    # Try environment variables
    env_value = os.environ.get(key_name)
    if env_value:
        return env_value, 'env'
    
    return None, 'none'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="AI Stock Report",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
    .report-container {
        background: #f8f9fa;
        padding: 30px;
        border-radius: 10px;
        line-height: 1.8;
    }
    .report-container h1 {
        color: #2c3e50;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .report-container h2 {
        color: #34495e;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .report-container p {
        margin-bottom: 15px;
    }
    .api-key-input {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Article Generation Prompt Template
ARTICLE_PROMPT_TEMPLATE = """You are writing a full Substack-style investment article.  
Tone: narrative, engaging, confident, analytically sharp, but accessible to non-experts.  
Use only verifiable, up-to-date public sources (SEC filings, investor presentations, earnings transcripts, reputable financial media).  
If any information is uncertain or not publicly disclosed, explicitly write: "N/A — not disclosed."

The only user input is this:
TICKER = "{ticker}"

Now generate the full article using the structure below.

------------------------------------------------------------
BEGIN ARTICLE
------------------------------------------------------------

1) THE HOOK  
Open with a compelling 1–3 sentence hook about why {ticker} matters *right now*.  
Set expectations for what the reader will learn.

2) THE BUSINESS — IN PLAIN ENGLISH  
Explain clearly:  
- What the company actually does  
- How it makes money  
- The real drivers of its stock price (pricing, volumes, regulation, margins, innovation, etc.)

3) THE 2025 STORY — WHAT HAPPENED AND WHY IT MATTERED  
Deliver a narrative, quarter-by-quarter walkthrough of 2025 events:  
- Key news, earnings, guidance, financing, regulatory issues  
- Each major event MUST have date + source  
- Explain why each event meaningfully changed sentiment or forward expectations  
- Write as a story, not a bullet list

4) 2026 CATALYSTS — WHAT COULD MOVE THE STOCK NEXT  
Provide a ranked catalyst list:  
For each catalyst:  
- Name  
- Expected timing  
- Why it matters  
- Who it impacts (investors, customers, regulators)  
- Label as **High Impact**, **Medium Impact**, or **Speculative**

5) GROWTH TRAJECTORY — 2026 TO 2028  
Start with a narrative of what needs to go right.  
Then provide a projection table:

| Year | Revenue | YoY % | EPS |
|------|---------|-------|-----|
| 2026 | X | X% | X |
| 2027 | X | X% | X |
| 2028 | X | X% | X |

Explain assumptions ("If margins normalize to X%…")  
Then provide Base / Bull / Bear outcomes (1–2 sentences each).

6) WHAT WALL STREET THINKS  
Present the consensus view:  
- Rating  
- Average price target  
- High and low targets  
- 2–3 recent analyst theses with dates  
If unavailable, say: "Consensus data N/A — not publicly available."

7) SECTOR WEATHER REPORT  
Describe the broader industry setup:  
- Macro or regulatory winds shaping the sector  
- Competition, pricing, demand trends  
- Why {ticker} is advantaged — or exposed — within this environment  
Write in simple, clear, non-jargon language.

8) THE TAKEAWAY (INVESTMENT VIEW)  
Give a clear verdict: **Buy / Hold / Sell**  
Provide:  
- 3 reasons supporting the call  
- 3 risks that could break the thesis  
Finish with a punchy one-line conclusion ("If 2026 goes right, this stock won't stay at $X for long.")

9) SOURCES  
List 5–7 specific sources with:  
Publication / Title / Date  
No vague references ("news article" not allowed).

------------------------------------------------------------
END ARTICLE
------------------------------------------------------------

FORMATTING RULES  
- Substack tone  
- 600–900 words  
- Short paragraphs (2–4 sentences)  
- Bold used sparingly for emphasis  
- No invented data  
- Any missing numbers must be labeled "N/A"
"""

def call_openai_api(api_key: str, ticker: str) -> str:
    """Call OpenAI API to generate article using v1.0+ client syntax"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        prompt = ARTICLE_PROMPT_TEMPLATE.format(ticker=ticker.upper())
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional investment analyst and writer who creates comprehensive, well-researched stock analysis articles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        return "❌ Error: OpenAI package not installed. Run: pip install openai>=1.0.0"
    except Exception as e:
        return f"❌ Error calling OpenAI API: {str(e)}"

def call_anthropic_api(api_key: str, ticker: str) -> str:
    """Call Anthropic Claude API to generate article"""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = ARTICLE_PROMPT_TEMPLATE.format(ticker=ticker.upper())
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
        
    except ImportError:
        return "❌ Error: Anthropic package not installed. Run: pip install anthropic>=0.18.0"
    except Exception as e:
        return f"❌ Error calling Anthropic API: {str(e)}"

# ===== HEADER =====
st.markdown("""
<div class="main-header">
    <h1>📰 AI Stock Report Generator</h1>
    <p>Generate comprehensive Substack-style investment articles powered by AI</p>
</div>
""", unsafe_allow_html=True)

# ===== API KEY CONFIGURATION =====
st.markdown("### 🔑 API Configuration")

# Get stored API keys
openai_stored_key, openai_source = get_api_key_from_secrets("OPENAI_API_KEY")
anthropic_stored_key, anthropic_source = get_api_key_from_secrets("ANTHROPIC_API_KEY")

col1, col2 = st.columns([1, 2])

with col1:
    llm_provider = st.selectbox(
        "Select LLM Provider",
        ["OpenAI (GPT-4)", "Anthropic (Claude)"],
        help="Choose your preferred AI model provider"
    )

# Determine which stored key to use based on provider
if "OpenAI" in llm_provider:
    stored_key = openai_stored_key
    key_source = openai_source
    key_name = "OPENAI_API_KEY"
else:
    stored_key = anthropic_stored_key
    key_source = anthropic_source
    key_name = "ANTHROPIC_API_KEY"

with col2:
    if stored_key:
        # Show indicator that key is loaded from secrets/env
        source_label = "Streamlit secrets" if key_source == 'secrets' else "environment variable"
        st.success(f"✅ API key loaded from {source_label}")
        api_key = stored_key
        # Optional: Allow override
        override_key = st.text_input(
            "Override API Key (optional)",
            type="password",
            placeholder="Enter to override stored key",
            help=f"Leave empty to use stored key from {source_label}"
        )
        if override_key:
            api_key = override_key
    else:
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your API key",
            help="Your API key is not stored and only used for this session"
        )

st.markdown("---")

# ===== TICKER INPUT =====
st.markdown("### 📊 Stock Analysis")
col_ticker, col_button = st.columns([2, 1])

with col_ticker:
    ticker = st.text_input(
        "Enter Stock Ticker",
        placeholder="e.g., NVDA, TSLA, AAPL",
        help="Enter the stock symbol you want to analyze"
    ).upper()

with col_button:
    st.write("")  # Spacing
    st.write("")  # Spacing
    generate_button = st.button("🚀 Generate Report", type="primary", width="stretch")

# Initialize session state
if 'generated_report' not in st.session_state:
    st.session_state.generated_report = None
if 'report_ticker' not in st.session_state:
    st.session_state.report_ticker = None

# ===== GENERATE REPORT =====
if generate_button:
    if not ticker:
        st.error("❌ Please enter a stock ticker")
    elif not api_key:
        st.error("❌ Please enter your API key")
    else:
        with st.spinner(f"🤖 Generating comprehensive investment report for {ticker}... This may take 30-60 seconds..."):
            try:
                if "OpenAI" in llm_provider:
                    report = call_openai_api(api_key, ticker)
                else:
                    report = call_anthropic_api(api_key, ticker)
                
                st.session_state.generated_report = report
                st.session_state.report_ticker = ticker
                st.success(f"✅ Report generated successfully for {ticker}!")
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
                logger.error(f"Report generation error: {e}", exc_info=True)

# ===== DISPLAY REPORT =====
if st.session_state.generated_report:
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button("📋 Copy to Clipboard", width="stretch"):
            st.write("Report copied! (Use browser copy function)")
    
    with col2:
        # Download as text file
        report_date = datetime.now().strftime("%Y%m%d")
        filename = f"{st.session_state.report_ticker}_report_{report_date}.txt"
        st.download_button(
            label="💾 Download Report",
            data=st.session_state.generated_report,
            file_name=filename,
            mime="text/plain",
            width="stretch"
        )
    
    with col3:
        if st.button("🔄 Generate New", width="stretch"):
            st.session_state.generated_report = None
            st.session_state.report_ticker = None
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display report
    st.markdown(f"""
    <div class="report-container">
        <h1>{st.session_state.report_ticker} Investment Analysis</h1>
        <p style="color: #7f8c8d; font-style: italic;">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        <hr style="margin: 20px 0;">
    """, unsafe_allow_html=True)
    
    # Display the report content
    st.markdown(st.session_state.generated_report)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===== INSTRUCTIONS =====
else:
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Configure API**
        - Select your LLM provider (OpenAI or Anthropic)
        - API keys auto-load from secrets/env vars
        - Or enter your API key manually
        
        **Step 2: Enter Ticker**
        - Type any valid stock symbol (e.g., NVDA, TSLA)
        - Click "Generate Report"
        - Wait 30-60 seconds for AI analysis
        """)
    
    with col2:
        st.markdown("""
        **What You'll Get:**
        - ✅ Compelling hook and business overview
        - ✅ 2025 quarterly event narrative
        - ✅ 2026 catalysts ranked by impact
        - ✅ 2026-2028 growth projections
        - ✅ Wall Street consensus view
        - ✅ Sector analysis and positioning
        - ✅ Buy/Hold/Sell recommendation
        - ✅ Verified sources cited
        """)
    
    st.markdown("---")
    
    st.info("""
    💡 **Pro Tip:** The AI generates 600-900 word Substack-style articles with narrative flow, 
    data-backed insights, and clear investment theses. Perfect for research, newsletters, or portfolio analysis.
    """)
    
    # API Key Help
    with st.expander("🔐 Where to get API keys?"):
        st.markdown("""
        **OpenAI (GPT-4):**
        1. Visit [platform.openai.com](https://platform.openai.com)
        2. Sign up or log in
        3. Go to API Keys section
        4. Create new secret key
        
        **Anthropic (Claude):**
        1. Visit [console.anthropic.com](https://console.anthropic.com)
        2. Sign up or log in
        3. Navigate to API Keys
        4. Generate new key
        
        ⚠️ **Important:** Keep your API keys secure and never share them publicly.
        """)
    
    # Environment Variable Setup Help
    with st.expander("⚙️ Setting up API keys for automatic loading"):
        st.markdown("""
        You can configure API keys to load automatically using one of these methods:
        
        **Option 1: Streamlit Secrets (Recommended for Streamlit Cloud)**
        1. Create a `.streamlit/secrets.toml` file in your project root
        2. Add your keys:
        ```toml
        OPENAI_API_KEY = "sk-your-openai-api-key"
        ANTHROPIC_API_KEY = "sk-ant-your-anthropic-api-key"
        ```
        3. Ensure `.streamlit/secrets.toml` is in your `.gitignore`
        
        **Option 2: Environment Variables (Recommended for local development)**
        
        *Linux/macOS:*
        ```bash
        export OPENAI_API_KEY="sk-your-openai-api-key"
        export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key"
        ```
        
        *Windows (PowerShell):*
        ```powershell
        $env:OPENAI_API_KEY="sk-your-openai-api-key"
        $env:ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key"
        ```
        
        *Or add to your `.env` file and use python-dotenv*
        
        **Priority Order:**
        1. Streamlit secrets (`st.secrets`)
        2. Environment variables
        3. Manual input in UI
        
        **Required Package Versions:**
        - `openai>=1.0.0`
        - `anthropic>=0.18.0`
        """)

# ===== FOOTER =====
st.markdown("---")
st.caption("📰 AI Stock Report Generator | Powered by Advanced LLMs | For informational purposes only - not financial advice")
