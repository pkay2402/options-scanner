"""
Thematic Stock Tracker
Track performance of stocks within major investment themes
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


# Define thematic baskets - EXPANDED UNIVERSE (150+ stocks across 20+ themes)
THEMES = {

    'AI Infrastructure': {
        'ARM': 'ARM Holdings - chip design',
        'SMCI': 'Super Micro - AI servers',
        'MRVL': 'Marvell - Data center chips',
        'ANET': 'Arista - AI networking'
    },

    'AI Software & Cloud': {
        'PLTR': 'Palantir - Enterprise AI',
        'NOW': 'ServiceNow - Enterprise AI',
        'CRM': 'Salesforce - Enterprise AI',
        'SNOW': 'Snowflake - Data cloud',
        'MDB': 'MongoDB - Database',
        'DDOG': 'Datadog - Observability'
    },

    'AI Applications': {
        'ADBE': 'Adobe - Generative AI monetization',
        'INTU': 'Intuit - AI for SMB finance',
        'DOCU': 'DocuSign - AI contract lifecycle',
        'PATH': 'UiPath - AI automation',
        'TEM': 'Tempus AI - Healthcare AI',
        'APP': 'AppLovin - AI ad optimization'
    },

    'Traditional Energy': {
        'XOM': 'ExxonMobil - Integrated major',
        'CVX': 'Chevron - Integrated major',
        'COP': 'ConocoPhillips - Pure upstream',
        'EOG': 'EOG Resources - Shale leader',
        'SLB': 'Schlumberger - Services',
        'OXY': 'Occidental - Buffett backed',
        'MPC': 'Marathon Petroleum - Refining',
        'VLO': 'Valero - Refining',
        'PSX': 'Phillips 66 - Refining',
        'HAL': 'Halliburton - Services',
        'BKR': 'Baker Hughes - Services',
        'DVN': 'Devon Energy - Shale',
        'FANG': 'Diamondback - Permian leader'
    },

    'Defense & Aerospace': {
        'LMT': 'Lockheed Martin - F-35, missiles',
        'RTX': 'Raytheon - Defense & aero',
        'NOC': 'Northrop Grumman - Bombers',
        'GD': 'General Dynamics - Ships',
        'BA': 'Boeing - Commercial & defense',
        'LHX': 'L3Harris - Defense electronics',
        'TXT': 'Textron - Military helicopters',
        'HWM': 'Howmet - Aerospace parts',
        'AXON': 'Axon - Law enforcement tech',
        'TDG': 'TransDigm - Aerospace aftermarket',
        'ANSS': 'Ansys - Defense simulation software',
        'SPR': 'Spirit AeroSystems - Aero structures'
    },

    'Government & Defense Software': {
        'BAH': 'Booz Allen - Defense AI consulting',
        'SAIC': 'Science Applications - Gov IT',
        'CACI': 'CACI - Signals & cyber',
        'LDOS': 'Leidos - Defense digital'
    },

    'Drones & eVTOL': {
        'AVAV': 'AeroVironment - Military drones',
        'KTOS': 'Kratos - Defense drones & targets',
        'IRDM': 'Iridium - Satellite comms',
        'ACHR': 'Archer Aviation - eVTOL',
        'JOBY': 'Joby Aviation - eVTOL air taxi',
        'LILM': 'Lilium - eVTOL jets',
        'EH': 'EHang - Autonomous aerial vehicles',
        'ONDS': 'ONDAS - Drone comms systems'
    },

    'GLP-1 Weight Loss': {
        'LLY': 'Eli Lilly - Mounjaro, Zepbound',
        'NVO': 'Novo Nordisk - Ozempic, Wegovy',
        'VKTX': 'Viking - Oral GLP-1',
        'RYTM': 'Rhythm - Obesity drugs'
    },

    'GLP-1 Beneficiaries': {
        'UNH': 'UnitedHealth - Insurance savings',
        'ELV': 'Elevance - Insurance savings',
        'CI': 'Cigna - Insurance savings',
        'HUM': 'Humana - Insurance savings',
        'ISRG': 'Intuitive Surgical - Elective surgery',
        'SYK': 'Stryker - Orthopedics',
        'ZBH': 'Zimmer Biomet - Joint replacement',
        'DGX': 'Quest Diagnostics - Health screening'
    },

    'Data Center & Power': {
        'EQIX': 'Equinix - Data center REIT',
        'DLR': 'Digital Realty - Data center REIT',
        'VST': 'Vistra - Power generation',
        'CEG': 'Constellation - Nuclear power',
        'NRG': 'NRG Energy - Power generation',
        'ETN': 'Eaton - Electrical equipment',
        'HUBB': 'Hubbell - Grid equipment',
        'AAON': 'AAON - HVAC & cooling',
        'VRT': 'Vertiv - Data center cooling',
        'PWR': 'Quanta Services - Grid infrastructure',
        'GEV': 'GE Vernova - Power equipment',
        'MOD': 'Modine - Thermal management',
        'BE': 'Bloom Energy - Fuel cells for data centers',
        'GWH': 'ESS Tech - Long-duration storage'
    },

    'Utilities & Grid Expansion': {
        'SO': 'Southern Company',
        'DUK': 'Duke Energy - Regulated utility',
        'EXC': 'Exelon - Nuclear & grid',
        'NEE': 'NextEra - Renewables & grid'
    },

    'Reshoring & Manufacturing': {
        'CAT': 'Caterpillar - Construction equipment',
        'DE': 'Deere - Industrial equipment',
        'PH': 'Parker Hannifin - Motion control',
        'ITW': 'Illinois Tool Works',
        'CMI': 'Cummins - Engines',
        'HON': 'Honeywell - Aerospace & buildings'
    },

    'Clean Energy': {
        'TSLA': 'Tesla - EVs & storage',
        'ENPH': 'Enphase - Solar inverters',
        'SEDG': 'SolarEdge - Solar equipment',
        'FSLR': 'First Solar - Solar panels',
        'ALB': 'Albemarle - Lithium',
        'SQM': 'SQM - Lithium',
        'LAC': 'Lithium Americas',
        'RIVN': 'Rivian - EV startup',
        'LCID': 'Lucid - EV luxury',
        'CHPT': 'ChargePoint - EV charging'
    },

    'Critical Minerals & Materials': {
        'FCX': 'Freeport-McMoRan - Copper',
        'TECK': 'Teck Resources - Copper & zinc',
        'MP': 'MP Materials - Rare earths',
        'BHP': 'BHP - Diversified miner',
        'RIO': 'Rio Tinto - Aluminum & copper'
    },

    'Semiconductors': {
        'NVDA': 'Nvidia - GPUs',
        'AMD': 'AMD - CPUs/GPUs',
        'INTC': 'Intel - CPUs',
        'TSM': 'TSMC - Foundry',
        'AVGO': 'Broadcom',
        'QCOM': 'Qualcomm',
        'TXN': 'Texas Instruments',
        'AMAT': 'Applied Materials',
        'ASML': 'ASML - Lithography',
        'LRCX': 'Lam Research',
        'KLAC': 'KLA Corp',
        'MU': 'Micron - Memory',
        'ADI': 'Analog Devices',
        'ON': 'ON Semi'
    },

    'Cybersecurity': {
        'CRWD': 'CrowdStrike',
        'PANW': 'Palo Alto Networks',
        'ZS': 'Zscaler',
        'FTNT': 'Fortinet',
        'S': 'SentinelOne',
        'OKTA': 'Okta',
        'CYBR': 'CyberArk'
    },

    'Fintech & Payments': {
        'V': 'Visa',
        'MA': 'Mastercard',
        'PYPL': 'PayPal',
        'XYZ': 'Block',
        'COIN': 'Coinbase',
        'SOFI': 'SoFi',
        'NU': 'Nu Holdings',
        'AFRM': 'Affirm',
        'HOOD': 'Robinhood - Retail trading'
    },

    'Capital Markets & Exchanges': {
        'ICE': 'Intercontinental Exchange',
        'CME': 'CME Group',
        'NDAQ': 'Nasdaq',
        'MKTX': 'MarketAxess'
    },

    'Biotech & Genomics': {
        'REGN': 'Regeneron',
        'VRTX': 'Vertex',
        'GILD': 'Gilead',
        'MRNA': 'Moderna',
        'ILMN': 'Illumina',
        'CRSP': 'CRISPR',
        'BEAM': 'Beam Therapeutics',
        'ALNY': 'Alnylam',
        'RXRX': 'Recursion'
    },

    'Space & Satellites': {
        'RKLB': 'Rocket Lab',
        'ASTS': 'AST SpaceMobile',
        'SPCE': 'Virgin Galactic',
        'VSAT': 'Viasat'
    },

    'E-commerce & Digital Advertising': {
        'SHOP': 'Shopify',
        'MELI': 'MercadoLibre',
        'TTD': 'Trade Desk',
        'PINS': 'Pinterest',
        'RBLX': 'Roblox - Gaming & metaverse ads'
    },

    'Streaming & Entertainment': {
        'NFLX': 'Netflix - Streaming leader',
        'DIS': 'Disney - Media & parks',
        'WBD': 'Warner Bros Discovery',
        'PARA': 'Paramount - Media',
        'SPOT': 'Spotify - Audio streaming',
        'ROKU': 'Roku - Streaming devices'
    },

    'Luxury & Consumer': {
        'CVNA': 'Carvana - Used car platform',
        'ABNB': 'Airbnb - Travel platform',
        'UBER': 'Uber - Rideshare & delivery',
        'LYFT': 'Lyft - Rideshare',
        'DASH': 'DoorDash - Food delivery',
        'BKNG': 'Booking - Travel',
        'EXPE': 'Expedia - Travel'
    },

    'Cloud Infrastructure': {
        'MSFT': 'Microsoft - Azure cloud',
        'AMZN': 'Amazon - AWS cloud',
        'GOOGL': 'Google - GCP cloud',
        'META': 'Meta - AI infrastructure',
        'ORCL': 'Oracle - Enterprise cloud',
        'IBM': 'IBM - Hybrid cloud',
        'NET': 'Cloudflare - Edge cloud',
        'FSLY': 'Fastly - CDN'
    },

    'Real Estate Tech': {
        'Z': 'Zillow',
        'OPEN': 'Opendoor',
        'CSGP': 'CoStar'
    },

    'Quantum Computing': {
        'IONQ': 'IonQ',
        'RGTI': 'Rigetti',
        'QUBT': 'Quantum Computing Inc'
    },

    'Cannabis': {
        'TLRY': 'Tilray',
        'CGC': 'Canopy Growth',
        'SNDL': 'Sundial',
        'MSOS': 'US Cannabis ETF'
    },

    'Luxury & Consumer': {
        'LVMUY': 'LVMH',
        'NKE': 'Nike',
        'LULU': 'Lululemon',
        'DECK': 'Deckers',
        'EL': 'Estee Lauder',
        'TPR': 'Tapestry',
        'RH': 'Restoration Hardware - Luxury home',
        'CVNA': 'Carvana - Used car platform'
    },

    'Infrastructure & Construction': {
        'VMC': 'Vulcan Materials',
        'MLM': 'Martin Marietta',
        'URI': 'United Rentals',
        'MTZ': 'MasTec'
    },

    'Consumer Tech & Travel': {
        'UBER': 'Uber - Rideshare & autonomous',
        'ABNB': 'Airbnb - Travel recovery',
        'LYFT': 'Lyft - Rideshare'
    },

    'Healthcare Services': {
        'DV': 'DaVita - Dialysis services',
        'LH': 'LabCorp - Diagnostic labs',
        'DVA': 'DaVita alternative ticker'
    },

    'Industrial Robotics & Automation': {
        'FANUY': 'FANUC - Industrial robots',
        'ROK': 'Rockwell Automation',
        'EMR': 'Emerson - Process automation'
    }
}



class ThemeTracker:
    """Track performance of thematic stock baskets"""
    
    def __init__(self):
        self.themes = THEMES
    
    def get_theme_performance(self, theme_name: str, period: str = "1mo") -> pd.DataFrame:
        """Get performance data for a specific theme"""
        if theme_name not in self.themes:
            print(f"Theme '{theme_name}' not found")
            return pd.DataFrame()
        
        tickers = list(self.themes[theme_name].keys())
        data = {}
        
        for ticker in tickers:
            try:
                df = yf.download(ticker, period=period, progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        close_col = df.columns[df.columns.get_level_values(0) == 'Close'][0]
                        data[ticker] = df[close_col]
                    elif 'Close' in df.columns:
                        data[ticker] = df['Close']
            except Exception as e:
                print(f"  ⚠️  Error fetching {ticker}: {e}")
                continue
        
        return pd.DataFrame(data)
    
    def calculate_theme_returns(self, theme_name: str, periods: Dict[str, int] = None) -> pd.Series:
        """Calculate returns for a theme over multiple periods"""
        if periods is None:
            periods = {'1W': 5, '1M': 21, '3M': 63, '6M': 126}
        
        data = self.get_theme_performance(theme_name, period='6mo')
        if data.empty:
            return pd.Series()
        
        returns = {}
        for period_name, days in periods.items():
            if days < len(data):
                period_returns = ((data.iloc[-1] / data.iloc[-days]) - 1) * 100
                returns[period_name] = period_returns.mean()  # Equal-weighted average
            else:
                returns[period_name] = np.nan
        
        return pd.Series(returns)
    
    def get_all_themes_performance(self, period: str = '1W') -> pd.DataFrame:
        """Get performance summary for all themes"""
        period_map = {'1W': 5, '1M': 21, '3M': 63, '6M': 126}
        days = period_map.get(period, 5)
        
        results = {}
        for theme_name in self.themes.keys():
            data = self.get_theme_performance(theme_name, period='6mo')
            if not data.empty and days < len(data):
                theme_return = ((data.iloc[-1] / data.iloc[-days]) - 1) * 100
                results[theme_name] = theme_return.mean()
        
        return pd.Series(results).sort_values(ascending=False)
    
    def get_top_stocks_in_theme(self, theme_name: str, period: str = '1W', n: int = 5) -> pd.DataFrame:
        """Get top performing stocks within a theme"""
        period_map = {'1W': 5, '1M': 21, '3M': 63}
        days = period_map.get(period, 5)
        
        data = self.get_theme_performance(theme_name, period='3mo')
        if data.empty or days >= len(data):
            return pd.DataFrame()
        
        returns = ((data.iloc[-1] / data.iloc[-days]) - 1) * 100
        top_stocks = returns.sort_values(ascending=False).head(n)
        
        # Add descriptions
        result = pd.DataFrame({
            'Return (%)': top_stocks.values,
            'Description': [self.themes[theme_name].get(ticker, '') for ticker in top_stocks.index]
        }, index=top_stocks.index)
        
        return result
    
    def print_theme_report(self, theme_name: str):
        """Print a detailed report for a theme"""
        if theme_name not in self.themes:
            print(f"Theme '{theme_name}' not found")
            return
        
        print("=" * 70)
        print(f"THEME REPORT: {theme_name}")
        print("=" * 70)
        
        # Get data
        data = self.get_theme_performance(theme_name, period='6mo')
        if data.empty:
            print("No data available")
            return
        
        # Calculate returns
        current = data.iloc[-1]
        week_ago = data.iloc[-5] if len(data) >= 5 else data.iloc[0]
        month_ago = data.iloc[-21] if len(data) >= 21 else data.iloc[0]
        
        weekly_returns = ((current / week_ago) - 1) * 100
        monthly_returns = ((current / month_ago) - 1) * 100
        
        # Print table
        print(f"\n{'Ticker':<8} {'Description':<35} {'1 Week':<10} {'1 Month':<10}")
        print("-" * 70)
        
        for ticker in sorted(weekly_returns.index, key=lambda x: weekly_returns[x], reverse=True):
            desc = self.themes[theme_name].get(ticker, '')
            week_ret = weekly_returns[ticker]
            month_ret = monthly_returns[ticker]
            
            print(f"{ticker:<8} {desc:<35} {week_ret:>8.2f}% {month_ret:>9.2f}%")
        
        # Theme average
        print("-" * 70)
        print(f"{'AVERAGE':<8} {'Theme equal-weighted':<35} {weekly_returns.mean():>8.2f}% {monthly_returns.mean():>9.2f}%")
        print("=" * 70)


def main():
    """Example usage and weekly theme report"""
    print("=" * 70)
    print("WEEKLY THEMATIC PERFORMANCE REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    tracker = ThemeTracker()
    
    # Overall theme performance
    print("\n📊 ALL THEMES - WEEKLY PERFORMANCE")
    print("-" * 70)
    all_themes = tracker.get_all_themes_performance(period='1W')
    for theme, ret in all_themes.items():
        status = "🔥" if ret > 2 else "🟢" if ret > 0 else "🔴"
        print(f"{status} {theme:<30s}: {ret:>6.2f}%")
    
    # Top 3 themes - detailed breakdown
    print("\n\n📈 TOP 3 THEMES - DETAILED BREAKDOWN")
    print("=" * 70)
    
    for theme in all_themes.head(3).index:
        print()
        tracker.print_theme_report(theme)
        print()
    
    # Bottom 2 themes
    print("\n\n📉 WEAKEST 2 THEMES - DETAILED BREAKDOWN")
    print("=" * 70)
    
    for theme in all_themes.tail(2).index:
        print()
        tracker.print_theme_report(theme)
        print()


if __name__ == "__main__":
    main()
