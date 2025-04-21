import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import uuid
import zipfile
import io
import shutil

def load_community_analysts():
    """
    Load community analysts data
    Returns an empty DataFrame if no data exists
    """
    try:
        if os.path.exists('data/community_analysts.csv') and os.path.getsize('data/community_analysts.csv') > 0:
            analysts_df = pd.read_csv('data/community_analysts.csv')
            return analysts_df
        else:
            # Create empty DataFrame with defined column structure
            return pd.DataFrame({
                'id': [],
                'name': [],
                'author': [],
                'contact': [],
                'description': [],
                'category': [],
                'upload_date': [],
                'file_path': [],
                'status': [],  # 'pending', 'approved', 'rejected'
                'rating': [],
                'users': [],
                'reviews_count': []
            })
    except Exception as e:
        st.warning(f"Error loading community analysts data: {e}")
        return pd.DataFrame()

def save_community_analysts(df):
    """Save community analysts data to CSV file"""
    try:
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/community_analysts.csv', index=False)
        return True
    except Exception as e:
        st.error(f"Error saving community analysts data: {e}")
        return False

def load_analyst_reviews(analyst_id):
    """
    Load reviews for a specific analyst
    Returns an empty DataFrame if no data exists
    """
    try:
        reviews_file = f'data/reviews/{analyst_id}.csv'
        if os.path.exists(reviews_file) and os.path.getsize(reviews_file) > 0:
            reviews_df = pd.read_csv(reviews_file)
            return reviews_df
        else:
            # Create empty DataFrame with defined column structure
            return pd.DataFrame({
                'user_id': [],
                'username': [],
                'rating': [],
                'comment': [],
                'date': []
            })
    except Exception as e:
        st.warning(f"Error loading analyst reviews data: {e}")
        return pd.DataFrame()

def save_analyst_review(analyst_id, user_id, username, rating, comment):
    """Save user review for an analyst"""
    try:
        # Ensure reviews directory exists
        os.makedirs('data/reviews', exist_ok=True)
        
        # Load existing reviews
        reviews_df = load_analyst_reviews(analyst_id)
        
        # Add new review
        new_review = pd.DataFrame({
            'user_id': [user_id],
            'username': [username],
            'rating': [rating],
            'comment': [comment],
            'date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        
        # Merge and save
        reviews_df = pd.concat([reviews_df, new_review], ignore_index=True)
        reviews_file = f'data/reviews/{analyst_id}.csv'
        reviews_df.to_csv(reviews_file, index=False)
        
        # Update analyst's overall rating
        update_analyst_rating(analyst_id)
        
        return True
    except Exception as e:
        st.error(f"Error saving review: {e}")
        return False

def update_analyst_rating(analyst_id):
    """Update analyst's overall rating"""
    try:
        # Load all analysts
        analysts_df = load_community_analysts()
        if analysts_df.empty or analyst_id not in analysts_df['id'].values:
            return False
        
        # Load reviews for this analyst
        reviews_df = load_analyst_reviews(analyst_id)
        if reviews_df.empty:
            return False
        
        # Calculate average rating
        avg_rating = reviews_df['rating'].mean()
        reviews_count = len(reviews_df)
        
        # Update analyst data
        analysts_df.loc[analysts_df['id'] == analyst_id, 'rating'] = round(avg_rating, 1)
        analysts_df.loc[analysts_df['id'] == analyst_id, 'reviews_count'] = reviews_count
        
        # Save updated data
        save_community_analysts(analysts_df)
        return True
    except Exception as e:
        st.error(f"Error updating rating: {e}")
        return False

def process_zip_upload(uploaded_file, name, author, contact, description, category):
    """Process user-uploaded ZIP file and add to community analyst list"""
    try:
        # Generate unique ID
        analyst_id = str(uuid.uuid4())
        
        # Create directory for saving files
        upload_dir = 'uploads/community_analysts'
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save ZIP file
        file_path = f"{upload_dir}/{analyst_id}.zip"
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Validate ZIP file structure (more detailed validation could be added here)
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Check if ZIP contents meet requirements
                file_list = zip_ref.namelist()
                if not file_list or not any(f.endswith('.py') for f in file_list):
                    os.remove(file_path)
                    return False, "ZIP file must contain at least one Python file"
        except:
            os.remove(file_path)
            return False, "Invalid ZIP file"
        
        # Add to analyst list
        analysts_df = load_community_analysts()
        new_analyst = pd.DataFrame({
            'id': [analyst_id],
            'name': [name],
            'author': [author],
            'contact': [contact],
            'description': [description],
            'category': [category],
            'upload_date': [datetime.now().strftime("%Y-%m-%d")],
            'file_path': [file_path],
            'status': ['pending'],  # Default to pending status
            'rating': [0.0],
            'users': [0],
            'reviews_count': [0]
        })
        
        # Merge and save
        analysts_df = pd.concat([analysts_df, new_analyst], ignore_index=True)
        save_community_analysts(analysts_df)
        
        return True, analyst_id
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return False, str(e)

def get_analyst_categories():
    """Get list of analyst categories"""
    return [
        "Technical Analysis",
        "Fundamental Analysis",
        "Sentiment Analysis",
        "Macroeconomic",
        "Risk Management",
        "Hybrid Strategy",
        "Other"
    ]

def increment_analyst_usage(analyst_id):
    """Increment usage count for an analyst"""
    try:
        analysts_df = load_community_analysts()
        if analysts_df.empty or analyst_id not in analysts_df['id'].values:
            return False
        
        # Increment usage count
        analysts_df.loc[analysts_df['id'] == analyst_id, 'users'] += 1
        
        # Save updated data
        save_community_analysts(analysts_df)
        return True
    except Exception as e:
        st.error(f"Error updating usage count: {e}")
        return False 