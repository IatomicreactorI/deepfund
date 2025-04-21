import streamlit as st
import pandas as pd
import os
import zipfile
from datetime import datetime
import io
import random
import uuid
import time
from .utils import (
    load_community_analysts, 
    save_community_analysts,
    load_analyst_reviews,
    save_analyst_review,
    process_zip_upload,
    get_analyst_categories,
    increment_analyst_usage
)

def display_community():
    """Display the Agent Community page"""
    st.divider()
    st.title("🌐 Agent Community - Analyst Hub")
    
    # Initialize session state
    if 'community_view' not in st.session_state:
        st.session_state.community_view = 'main'  # 'main', 'detail', 'upload_guide'
    
    if 'selected_analyst_id' not in st.session_state:
        st.session_state.selected_analyst_id = None
        
    if 'user_id' not in st.session_state:
        # Generate a temporary ID for the current user
        st.session_state.user_id = str(uuid.uuid4())
        
    if 'username' not in st.session_state:
        # Generate a random nickname
        adjectives = ["Happy", "Clever", "Curious", "Brave", "Friendly", "Patient"]
        animals = ["Panda", "Tiger", "Lion", "Elephant", "Giraffe", "Monkey", "Owl"]
        st.session_state.username = f"{random.choice(adjectives)}{random.choice(animals)}{random.randint(1, 999)}"
    
    # Navigation function
    def set_view(view, analyst_id=None):
        st.session_state.community_view = view
        if analyst_id is not None:
            st.session_state.selected_analyst_id = analyst_id
    
    # Display content based on current view
    if st.session_state.community_view == 'main':
        display_main_view(set_view)
    elif st.session_state.community_view == 'detail':
        display_analyst_detail(set_view)
    elif st.session_state.community_view == 'upload_guide':
        display_upload_guide(set_view)
    else:
        set_view('main')  # Default back to main page

def display_main_view(set_view):
    """Display the community main page"""
    st.markdown("""
    Welcome to the **Agent Community**! This is where you'll find a collection of financial analyst models created by community members.
    You can browse, use, rate these analysts, and even contribute your own designed analysts.
    """)
    
    # Upload section
    with st.expander("📤 Upload Your Analyst", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("Share your expertise by contributing your own analyst model with the community!")
        with col2:
            st.button("📝 View Design Guide", on_click=lambda: set_view('upload_guide'), use_container_width=True)
        
        # Upload form
        with st.form("analyst_upload_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Analyst Name", max_chars=50)
                author = st.text_input("Author", max_chars=50)
            with col2:
                contact = st.text_input("Contact Info (Email/Social Media)", max_chars=100)
                category = st.selectbox("Category", options=get_analyst_categories())
                
            description = st.text_area("Description (Explain your analyst's functionality and features)", max_chars=500)
            uploaded_file = st.file_uploader("Upload Analyst Code (ZIP format)", type=['zip'])
            
            submit = st.form_submit_button("Submit Analyst")
            if submit:
                if not name or not author or not contact or not description or not uploaded_file:
                    st.error("Please fill in all required fields and upload a file")
                else:
                    # Process upload
                    success, message = process_zip_upload(
                        uploaded_file, name, author, contact, description, category
                    )
                    if success:
                        st.success("Upload successful! Your analyst will be published after review.")
                    else:
                        st.error(f"Upload failed: {message}")
    
    st.divider()
    
    # Search and filter section
    st.subheader("🔍 Find Analysts")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("Search analyst names or descriptions", placeholder="Enter keywords...")
    with col2:
        filter_option = st.selectbox(
            "Sort by", 
            ["Highest Rated", "Most Popular", "Newest"]
        )
    
    category_filter = st.multiselect(
        "Filter by category", 
        options=get_analyst_categories()
    )
    
    # Load analyst data
    analysts_df = load_community_analysts()
    
    # Filter to show only approved analysts
    analysts_df = analysts_df[analysts_df['status'] == 'approved'].copy() if not analysts_df.empty else analysts_df
    
    # Apply search and filters
    if not analysts_df.empty:
        # Search
        if search_term:
            search_mask = (
                analysts_df['name'].str.contains(search_term, case=False, na=False) | 
                analysts_df['description'].str.contains(search_term, case=False, na=False) |
                analysts_df['author'].str.contains(search_term, case=False, na=False)
            )
            analysts_df = analysts_df[search_mask]
        
        # Category filter
        if category_filter:
            analysts_df = analysts_df[analysts_df['category'].isin(category_filter)]
        
        # Sort
        if filter_option == "Highest Rated":
            analysts_df = analysts_df.sort_values(by='rating', ascending=False)
        elif filter_option == "Most Popular":
            analysts_df = analysts_df.sort_values(by='users', ascending=False)
        elif filter_option == "Newest":
            analysts_df = analysts_df.sort_values(by='upload_date', ascending=False)
    
    # Display analyst list
    if analysts_df.empty:
        st.info("No community analysts available yet. Be the first to contribute!")
    else:
        st.subheader("📊 Community Analysts")
        
        # Create a nicer table interface
        for _, analyst in analysts_df.iterrows():
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    # Display rating and usage data
                    st.metric(
                        "Rating", 
                        f"{analyst['rating']:.1f}/5.0" if pd.notna(analyst['rating']) else "No ratings",
                        help="Average user rating"
                    )
                    st.caption(f"📊 Users: {analyst['users']}")
                    st.caption(f"✍️ Reviews: {analyst['reviews_count']}")
                
                with col2:
                    # Display name and description
                    st.markdown(f"### {analyst['name']}")
                    st.caption(f"Author: {analyst['author']} | Category: {analyst['category']} | Uploaded: {analyst['upload_date']}")
                    st.markdown(f"{analyst['description']}")
                
                with col3:
                    # View details button
                    st.write("")  # Add some space
                    st.button(
                        "View Details", 
                        key=f"view_{analyst['id']}", 
                        on_click=lambda aid=analyst['id']: set_view('detail', aid),
                        use_container_width=True
                    )
                    st.write("")  # Add some space
                    st.button(
                        "Add to My Agent", 
                        key=f"add_{analyst['id']}", 
                        on_click=lambda aid=analyst['id']: increment_analyst_usage(aid),
                        use_container_width=True
                    )

def display_analyst_detail(set_view):
    """Display details for a single analyst"""
    # Get the selected analyst ID
    analyst_id = st.session_state.selected_analyst_id
    if not analyst_id:
        st.error("Analyst information not found")
        st.button("Return to List", on_click=lambda: set_view('main'))
        return
    
    # Load analyst data
    analysts_df = load_community_analysts()
    if analysts_df.empty or analyst_id not in analysts_df['id'].values:
        st.error("Analyst information not found")
        st.button("Return to List", on_click=lambda: set_view('main'))
        return
    
    # Get analyst info
    analyst = analysts_df[analysts_df['id'] == analyst_id].iloc[0]
    
    # Return button
    st.button("← Back to List", on_click=lambda: set_view('main'))
    
    # Display analyst details
    st.title(analyst['name'])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.caption(f"Author: {analyst['author']} | Category: {analyst['category']} | Uploaded: {analyst['upload_date']}")
        st.markdown(f"### Description")
        st.markdown(analyst['description'])
    
    with col2:
        # Rating and statistics
        rating_display = f"{analyst['rating']:.1f}/5.0" if pd.notna(analyst['rating']) and analyst['rating'] > 0 else "No ratings"
        st.metric("Rating", rating_display)
        st.metric("Users", analyst['users'])
        st.metric("Reviews", analyst['reviews_count'])
        
        # Add to my agent button
        st.button(
            "Add to My Agent", 
            key="add_to_agent", 
            on_click=lambda: increment_analyst_usage(analyst_id),
            use_container_width=True,
            type="primary"
        )
    
    st.divider()
    
    # Display review section
    st.subheader("💬 User Reviews")
    
    # Add review form
    with st.form("add_review_form"):
        st.markdown("### Add Your Review")
        rating = st.slider("Rating", min_value=1, max_value=5, value=5, step=1)
        comment = st.text_area("Comment (Optional)", max_chars=500)
        
        submitted = st.form_submit_button("Submit Review")
        if submitted:
            # Save review
            success = save_analyst_review(
                analyst_id, 
                st.session_state.user_id,
                st.session_state.username,
                rating,
                comment
            )
            if success:
                st.success("Review submitted!")
                # Show loading effect, then refresh
                with st.spinner("Updating..."):
                    time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to submit review")
    
    # Display existing reviews
    reviews_df = load_analyst_reviews(analyst_id)
    
    if reviews_df.empty:
        st.info("No reviews yet. Be the first to review!")
    else:
        # Sort by date, newest first
        reviews_df = reviews_df.sort_values(by='date', ascending=False)
        
        for _, review in reviews_df.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    # Rating and user info
                    st.markdown(f"### {'⭐' * int(review['rating'])}")
                    st.caption(f"Rating: {review['rating']}/5")
                
                with col2:
                    # Review content
                    st.markdown(f"**{review['username']}** · {review['date']}")
                    if review['comment']:
                        st.markdown(review['comment'])
                    else:
                        st.caption("(No comment provided)")

def display_upload_guide(set_view):
    """Display the upload guide"""
    st.button("← Back to Community", on_click=lambda: set_view('main'))
    
    st.title("📝 Analyst Design & Upload Guide")
    
    st.markdown("""
    ## What is an Analyst?
    
    In the DeepFund platform, an **Analyst** is an AI component focused on financial market analysis. Each analyst has a specific area of expertise, 
    such as technical analysis, fundamental analysis, news sentiment analysis, etc. Multiple analysts can work together to provide comprehensive market insights for an AI agent.
    
    ## Design Guidelines
    
    ### Basic Requirements
    
    1. **Python Format**: All analysts must be written in Python and follow PEP 8 coding standards
    2. **Main Function**: Must include a primary analysis function that receives standardized market data and returns analysis results
    3. **No External Dependencies**: Avoid using uncommon third-party libraries (standard library and common data science libraries are fine)
    4. **Documentation**: Code must include clear comments and function documentation
    
    ### File Structure
    
    Your uploaded ZIP file should contain the following:
    
    ```
    analyst_name/
    ├── __init__.py       # Package initialization file
    ├── analyst.py        # Main analysis logic
    ├── requirements.txt  # Dependency list (if any)
    └── README.md         # Usage documentation
    ```
    
    ### Interface Specification
    
    Each analyst must implement the following standard interface function:
    
    ```python
    def analyze(market_data, time_period='daily', **kwargs):
        '''
        Analyze market data and return results
        
        Parameters:
            market_data (DataFrame): Market data, including OHLCV and other indicators
            time_period (str): Time period ('daily', 'weekly', 'monthly')
            **kwargs: Additional parameters
            
        Returns:
            dict: Dictionary containing analysis results
        '''
        # Implement your analysis logic
        pass
    ```
    
    ### Return Value Format
    
    The analysis function should return a dictionary with the following keys:
    
    ```python
    {
        'recommendation': str,  # 'buy', 'sell', 'hold'
        'confidence': float,    # Confidence level (0.0-1.0)
        'reasoning': str,       # Analysis reasoning process
        'metrics': dict         # Related metrics and data points
    }
    ```
    
    ## Review Process
    
    1. **Submission**: Upload your analyst ZIP file and fill in the required information
    2. **Automatic Validation**: The system will check file structure and basic interface compatibility
    3. **Manual Review**: Our team will evaluate your analyst's quality and security
    4. **Testing**: The analyst will be tested in a simulated environment to verify performance
    5. **Publication**: Approved analysts will be added to the community list
    
    The review process typically takes 3-5 business days. We particularly focus on code quality, documentation completeness, and the reasonableness of the analysis method.
    
    ## Example Code
    
    Here's a simple technical analyst example:
    
    ```python
    import pandas as pd
    import numpy as np
    
    def calculate_rsi(data, window=14):
        '''Calculate Relative Strength Index (RSI)'''
        diff = data.diff()
        gain = diff.where(diff > 0, 0).rolling(window=window).mean()
        loss = -diff.where(diff < 0, 0).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def analyze(market_data, time_period='daily', **kwargs):
        '''RSI Analyst main function'''
        # Parameter setup
        rsi_window = kwargs.get('rsi_window', 14)
        overbought = kwargs.get('overbought', 70)
        oversold = kwargs.get('oversold', 30)
        
        # Ensure market data contains closing prices
        if 'close' not in market_data.columns:
            return {
                'recommendation': 'hold',
                'confidence': 0.0,
                'reasoning': 'Missing closing prices in data',
                'metrics': {}
            }
        
        # Calculate RSI
        rsi = calculate_rsi(market_data['close'], window=rsi_window)
        current_rsi = rsi.iloc[-1]
        
        # Make recommendation based on RSI
        if current_rsi < oversold:
            recommendation = 'buy'
            confidence = min(1.0, 2.0 * (oversold - current_rsi) / oversold)
            reasoning = f'RSI ({current_rsi:.2f}) is below oversold level ({oversold})'
        elif current_rsi > overbought:
            recommendation = 'sell'
            confidence = min(1.0, (current_rsi - overbought) / (100 - overbought))
            reasoning = f'RSI ({current_rsi:.2f}) is above overbought level ({overbought})'
        else:
            recommendation = 'hold'
            # Lower confidence when in neutral zone, more so in the middle
            mid_point = (overbought + oversold) / 2
            distance = abs(current_rsi - mid_point)
            range_half = (overbought - oversold) / 2
            confidence = distance / range_half
            reasoning = f'RSI ({current_rsi:.2f}) is in neutral zone'
        
        # Return results
        return {
            'recommendation': recommendation,
            'confidence': min(0.95, confidence),  # Limit maximum confidence
            'reasoning': reasoning,
            'metrics': {
                'rsi': current_rsi,
                'rsi_window': rsi_window,
                'overbought': overbought,
                'oversold': oversold
            }
        }
    ```
    
    ## Contact Support
    
    If you have difficulty designing or uploading an analyst, please contact our support team: support@deepfund.ai
    
    Happy coding!
    """)
    
    st.button("I understand, back to upload", on_click=lambda: set_view('main')) 