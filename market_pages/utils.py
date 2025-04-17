import streamlit as st

def load_subpages():
    """加载所有市场子页面模块
    
    Returns:
        tuple: (sub_pages字典, imports_successful布尔值)
    """
    # --- Import Sub-page Modules ---
    try:
        from . import cs2_skins
        from . import crypto
        from . import gold
        from . import oil
        from . import renewables
        from . import us_stocks_full
        
        sub_pages = {
            'cs2_skins.py': cs2_skins.display_cs2_skins_market,
            'crypto.py': crypto.display_crypto_market,
            'gold.py': gold.display_gold_market,
            'oil.py': oil.display_oil_market,
            'renewables.py': renewables.display_renewables_market,
            'us_stocks_full.py': us_stocks_full.display_us_stocks_full_market
        }
        imports_successful = True
    except ImportError as e:
        st.error(f"Failed to import market sub-pages. Ensure 'market_pages/__init__.py' exists. Error: {e}")
        imports_successful = False
        sub_pages = {}
        
    return sub_pages, imports_successful

def get_market_hub_data():
    """获取市场中心数据
    
    Returns:
        list: 市场数据列表
    """
    # Define market sectors (Ensure keys match the sub_pages dict)
    markets_hub = [
        {'name': 'Gold', 'icon': '🧈', 'file': 'gold.py', 'desc': 'Precious metals analysis and forecasts.'},
        {'name': 'Oil', 'icon': '🛢️', 'file': 'oil.py', 'desc': 'Energy market trends and crude oil data.'},
        {'name': 'Cryptocurrencies', 'icon': '₿', 'file': 'crypto.py', 'desc': 'Digital asset insights and blockchain news.'},
        {'name': 'US Stocks', 'icon': '📈', 'file': 'us_stocks_full.py', 'desc': 'Comprehensive US stock market data.'},
        {'name': 'Renewable Energy', 'icon': '🔋', 'file': 'renewables.py', 'desc': 'Green energy sector performance.'},
        {'name': 'CS2 Skins Trading', 'icon': '🌈', 'file': 'cs2_skins.py', 'desc': 'Virtual item market analysis (CS2).'}
    ]
    
    return markets_hub 