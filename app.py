import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved models and data
@st.cache_resource
def load_data():
    try:
        import os
        expected = ['top_50_books.pkl', 'similarity_scores.pkl', 'pt.pkl', 'books_data.pkl']
        for f in expected:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing file: {f}")
        top_books = joblib.load('top_50_books.pkl')
        similarity = joblib.load('similarity_scores.pkl')
        pt = joblib.load('pt.pkl')
        books_data = joblib.load('books_data.pkl')
        return top_books, similarity, pt, books_data
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        return None, None, None, None

# Function to clean and convert year data
def clean_year_data(year_series):
    """Convert year data to numeric, handling errors"""
    # Convert to numeric, coercing errors to NaN
    year_series = pd.to_numeric(year_series, errors='coerce')
    # Remove unrealistic years (before 1000 or after current year + 1)
    current_year = pd.Timestamp.now().year
    year_series = year_series[(year_series >= 1000) & (year_series <= current_year + 1)]
    return year_series

# Function to calculate average rating
def calculate_average_rating(books_data, book_title):
    """Calculate average rating for a book (dataset ratings may be 1-10)."""
    if books_data is None or not isinstance(books_data, pd.DataFrame):
        return "N/A"
    book_ratings = books_data.loc[books_data['Book-Title'] == book_title, 'Book-Rating']
    if book_ratings.empty:
        return "No ratings"
    numeric_ratings = pd.to_numeric(book_ratings, errors='coerce').dropna()
    if numeric_ratings.empty:
        return "No ratings"
    return round(float(numeric_ratings.mean()), 1)

# cache raw image bytes to avoid re-downloading
@st.cache_data(show_spinner=False, max_entries=128, ttl=60*60*24)
def load_image_bytes(url: str):
    if not url or not isinstance(url, str) or "http" not in url:
        return None
    url = url.strip()
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.3, status_forcelist=(500, 502, 504))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    try:
        resp = session.get(url, timeout=8, headers=headers)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '')
        if 'image' not in content_type:
            return None
        return resp.content
    except Exception:
        return None

def load_image_from_url(url):
    """Return PIL.Image or None (wrapper around cached bytes)."""
    try:
        img_bytes = load_image_bytes(url)
        if img_bytes:
            return Image.open(io.BytesIO(img_bytes))
    except Exception:
        return None
    return None

# Function to get the best available image URL
def get_best_image_url(book_info):
    """Get the best available image URL from the book info"""
    if book_info is None:
        return None
    
    # Try different image URL columns in order of preference
    image_urls = [
        book_info.get('Image-URL-L'),
        book_info.get('Image-URL-M'), 
        book_info.get('Image-URL-S'),
        book_info.get('Image-URL-L', ''),
        book_info.get('Image-URL-M', ''),
        book_info.get('Image-URL-S', '')
    ]
    
    for url in image_urls:
        if url and pd.notna(url) and isinstance(url, str) and url.startswith('http'):
            # Fix common URL issues
            url = url.strip()
            if 'http://images.amazon.com/images/P/' in url:
                # Ensure it has the proper format
                if not url.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    url += '.jpg'  # Amazon images usually are jpg
            return url
    
    return None

# Function to get book details by title
def get_book_details(book_title, books_data):
    if books_data is None or not isinstance(books_data, pd.DataFrame):
        return None
    
    # Filter books by title and get the first occurrence
    book_matches = books_data[books_data['Book-Title'] == book_title]
    if len(book_matches) > 0:
        book_info = book_matches.iloc[0].copy()
        # Calculate average rating
        book_info['Average-Rating'] = calculate_average_rating(books_data, book_title)
        return book_info
    return None

# Recommendation function
def recommend(book_name, pt, similarity):
    try:
        if pt is None or similarity is None:
            return []
        # robust lookup for index
        if book_name not in pt.index:
            return []
        index = int(np.where(pt.index == book_name)[0][0])
        sim_row = similarity[index]
        if len(sim_row) != similarity.shape[1] and similarity.ndim == 2:
            # fallback if shape mismatch
            sim_row = np.asarray(sim_row).ravel()
        sorted_list = sorted(list(enumerate(sim_row)), key=lambda x: x[1], reverse=True)
        # skip the first match (itself), get next 5
        recommendations = [pt.index[i] for i, _ in sorted_list if pt.index[i] != book_name][:5]
        return recommendations
    except Exception:
        return []

# Improved display book card component
def display_book_card(book_info, card_title="Book"):
    if book_info is None:
        st.warning("No book information available")
        return
    
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image_url = get_best_image_url(book_info)
            image = load_image_from_url(image_url) if image_url else None
            if image:
                image.thumbnail((200, 300))
                st.image(image, use_container_width=True, caption=book_info['Book-Title'])
            else:
                # show placeholder; keep URL as caption (short)
                caption = image_url if image_url else "No image URL available"
                if isinstance(caption, str) and len(caption) > 100:
                    caption = caption[:97] + "..."
                st.image("https://via.placeholder.com/150x200/4B5563/FFFFFF?text=No+Image", use_container_width=True, caption=caption)
        
        with col2:
            st.subheader(book_info['Book-Title'])
            st.write(f"**Author:** {book_info.get('Book-Author', 'Unknown')}")
            # Year
            year = book_info.get('Year-Of-Publication', None)
            try:
                year_int = int(year)
                if 1000 <= year_int <= pd.Timestamp.now().year + 1:
                    st.write(f"**Year:** {year_int}")
                else:
                    st.write("**Year:** Unknown")
            except Exception:
                st.write("**Year:** Unknown")
            
            st.write(f"**Publisher:** {book_info.get('Publisher', 'Unknown')}")
            
            # Average rating display: dataset uses 1-10 scale; map to 5-star
            avg_rating = book_info.get('Average-Rating', 'N/A')
            if isinstance(avg_rating, (int, float)):
                # convert 1-10 to 0-5 scale
                rating_5 = min(5, max(0, avg_rating / 2.0))
                full_stars = int(rating_5)
                half_star = 1 if (rating_5 - full_stars) >= 0.5 else 0
                empty_stars = 5 - full_stars - half_star
                stars = "⭐" * full_stars + ("⯨" if half_star else "") + "☆" * empty_stars
                st.write(f"**Average Rating:** {stars} ({avg_rating}/10)")
            else:
                st.write(f"**Average Rating:** {avg_rating}")
            
        st.markdown("---")

# Debug function to check image URLs for a book
def debug_image_urls(book_title, books_data):
    """Debug function to check what image URLs are available"""
    book_info = get_book_details(book_title, books_data)
    if book_info is not None:
        st.write("### Debug Information")
        st.write(f"**Book:** {book_title}")
        st.write(f"Image-URL-L: `{book_info.get('Image-URL-L')}`")
        st.write(f"Image-URL-M: `{book_info.get('Image-URL-M')}`")
        st.write(f"Image-URL-S: `{book_info.get('Image-URL-S')}`")
        
        best_url = get_best_image_url(book_info)
        st.write(f"**Selected URL:** `{best_url}`")
        
        if best_url:
            image = load_image_from_url(best_url)
            if image:
                st.write("✅ Image loaded successfully!")
                st.image(image, caption="Debug Image", width=200)
            else:
                st.write("❌ Failed to load image")

# Top 50 Books Page
def top_books_page(top_books, books_data):
    st.title("📊 Top 50 Books")
    st.markdown("Explore the most popular books in our collection!")
    
    # Debug option
    if st.checkbox("Show debug information"):
        if books_data is not None:
            sample_book = books_data.iloc[0]['Book-Title'] if len(books_data) > 0 else "Unknown"
            debug_image_urls(sample_book, books_data)
    
    if top_books is not None:
        # Display the top books in a grid layout
        if isinstance(top_books, pd.DataFrame):
            # If top_books is a DataFrame, use the book titles from it
            book_titles = top_books['Book-Title'].head(50) if 'Book-Title' in top_books.columns else top_books.head(50)
        else:
            # If it's a list or array
            book_titles = top_books[:50]
        
        # Filter to unique titles
        unique_titles = pd.Series(book_titles).drop_duplicates().head(50)
        
        st.subheader(f"🎯 Top {len(unique_titles)} Books")
        
        # Display books in a grid
        books_per_row = 3
        for i in range(0, len(unique_titles), books_per_row):
            cols = st.columns(books_per_row)
            for j, col in enumerate(cols):
                if i + j < len(unique_titles):
                    book_title = unique_titles.iloc[i + j]
                    book_info = get_book_details(book_title, books_data)
                    if book_info is not None:
                        with col:
                            display_book_card(book_info, f"Top {i + j + 1}")
    
    # Additional book statistics
    if books_data is not None and isinstance(books_data, pd.DataFrame):
        st.subheader("📈 Collection Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_books = books_data['Book-Title'].nunique()
            st.metric("Unique Books", total_books)
        
        with col2:
            if 'Book-Author' in books_data.columns:
                unique_authors = books_data['Book-Author'].nunique()
                st.metric("Unique Authors", unique_authors)
            else:
                st.metric("Books in Top 50", min(50, len(unique_titles)) if 'unique_titles' in locals() else 50)
        
        with col3:
            if 'Year-Of-Publication' in books_data.columns:
                # Clean year data before calculating min
                clean_years = clean_year_data(books_data['Year-Of-Publication'])
                if len(clean_years) > 0:
                    min_year = int(clean_years.min())
                    st.metric("Earliest Publication", min_year)
                else:
                    st.metric("Earliest Publication", "N/A")
        
        with col4:
            if 'Year-Of-Publication' in books_data.columns:
                # Clean year data before calculating max
                clean_years = clean_year_data(books_data['Year-Of-Publication'])
                if len(clean_years) > 0:
                    max_year = int(clean_years.max())
                    st.metric("Latest Publication", max_year)
                else:
                    st.metric("Latest Publication", "N/A")

# Recommendation Page
def recommendation_page(pt, similarity, books_data):
    st.title("🔍 Book Recommendations")
    st.markdown("Get personalized book recommendations based on your preferences!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Find Your Next Read")
        
        # Get list of available books from the pivot table index
        book_list = pt.index.tolist() if pt is not None else []
        
        if book_list:
            # Book selection with search
            selected_book = st.selectbox(
                "Select a book you like:",
                options=book_list,
                index=0,
                help="Choose a book to get recommendations based on it"
            )
            
            # Debug selected book
            if st.checkbox("Debug selected book"):
                debug_image_urls(selected_book, books_data)
            
            # Show selected book details
            if selected_book:
                book_info = get_book_details(selected_book, books_data)
                if book_info is not None:
                    st.subheader("Selected Book")
                    display_book_card(book_info)
            
            # Recommendation button
            if st.button("Get Recommendations", type="primary", use_container_width=True):
                if selected_book:
                    with st.spinner("Finding similar books..."):
                        recommendations = recommend(selected_book, pt, similarity)
                    
                    if recommendations:
                        st.session_state.recommendations = recommendations
                        st.success(f"Found {len(recommendations)} recommendations!")
                    else:
                        st.error("Could not generate recommendations for this book. Please try another selection.")
                else:
                    st.warning("Please select a book first.")
        else:
            st.error("No books available for recommendation. Please check your data files.")
    
    with col2:
        # Display recommendations if available
        if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
            st.subheader("📖 Your Recommendations")
            
            for i, book_title in enumerate(st.session_state.recommendations, 1):
                book_info = get_book_details(book_title, books_data)
                if book_info is not None:
                    # Create an expander for each recommendation
                    with st.expander(f"#{i} {book_title}", expanded=True):
                        display_book_card(book_info)
        
        else:
            st.subheader("How It Works")
            st.markdown("""
            ### 📖 About the Recommendation System
            
            This system uses **content-based filtering** to suggest books similar to ones you already enjoy.
            
            **How it works:**
            - **Cosine Similarity**: The algorithm measures how similar books are based on their features
            - **Content Analysis**: Books are compared based on their characteristics and content
            - **Personalized Results**: Each recommendation is tailored to your specific selection
            
            **To get recommendations:**
            1. Select a book you enjoy from the dropdown menu
            2. Click the "Get Recommendations" button
            3. Discover 5 books that are similar to your selection
            """)

# Main app with navigation
def main():
    # Initialize session state for recommendations
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    
    # Load data
    with st.spinner("Loading book data..."):
        top_books, similarity, pt, books_data = load_data()
    
    if top_books is None:
        st.error("Could not load the necessary data files. Please make sure all .pkl files are in the correct directory.")
        return
    
    # Sidebar navigation
    st.sidebar.title("📚 Book Recommendation System")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.radio("Navigate to:", 
                           ["Home", "Top Books", "Recommendations"])
    
    # Home page
    if page == "Home":
        st.title("Welcome to the Book Recommendation System! 📚")
        st.markdown("""
        ## Discover Your Next Favorite Book
        
        This application helps you find books you'll love based on your reading preferences.
        Explore our collection, get personalized recommendations, and discover new authors!
        """)
        
        # Quick stats on home page
        if books_data is not None:
            st.subheader("📈 Quick Stats")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_books = books_data['Book-Title'].nunique()
                st.metric("Unique Books", total_books)
            
            with col2:
                if 'Book-Author' in books_data.columns:
                    unique_authors = books_data['Book-Author'].nunique()
                    st.metric("Authors", unique_authors)
                else:
                    st.metric("Available for Recommendation", len(pt.index) if pt is not None else "N/A")
            
            with col3:
                if 'Year-Of-Publication' in books_data.columns:
                    clean_years = clean_year_data(books_data['Year-Of-Publication'])
                    if len(clean_years) > 0:
                        min_year = int(clean_years.min())
                        st.metric("Earliest Book", min_year)
                    else:
                        st.metric("Earliest Book", "N/A")
            
            with col4:
                st.metric("Recommendation Engine", "Active ✓")
        
        # Sample book display
        if books_data is not None:
            st.subheader("🌟 Featured Books")
            # Get unique books
            sample_titles = books_data['Book-Title'].unique()[:3]
            for title in sample_titles:
                book_info = get_book_details(title, books_data)
                if book_info is not None:
                    display_book_card(book_info)
    
    # Top Books page
    elif page == "Top Books":
        top_books_page(top_books, books_data)
    
    # Recommendations page
    elif page == "Recommendations":
        recommendation_page(pt, similarity, books_data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("This book recommendation system uses advanced algorithms to suggest books you'll love based on your reading preferences.")

# Run the app
if __name__ == "__main__":
    main()