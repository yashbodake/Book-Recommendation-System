[Check out the live demo of the Book Recommendation System here!](https://yash-book-recommendation-system.streamlit.app/)

# Book Recommendation System
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue?logo=streamlit)](https://yash-book-recommendation-system.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://www.python.org/)
An interactive Streamlit app that helps discover books similar to favorites using content-based filtering with precomputed similarity scores and a pivot table of titles. The underlying data is sourced from the [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset), which includes Books.csv, Ratings.csv, and Users.csv files. Exploratory Data Analysis (EDA) and preprocessing steps—such as loading datasets, handling missing values, computing average ratings, generating top books, creating a pivot table, and calculating similarity scores—are performed in the accompanying Jupyter notebook (eda.ipynb) to prepare the .pkl artifacts for the app.
## Features
- **Top 50 showcase** with rich book cards, cover images, author, publisher, year validation, and a 5-star visualization mapped from a 1-10 rating scale
- **Recommendation workflow**: select a known book, generate 5 similar titles using cosine-style similarity over a pivot index, and browse details in expandable cards
- **Image handling** with retries, content-type checks, in-memory caching, and graceful fallbacks to a placeholder when covers are missing or invalid
- **Clean stats**: unique books/authors, earliest/latest publication year (with sanity filtering), and quick app health indicators
## App Structure
- Streamlit multipage-style navigation in a single file: Home, Top Books, Recommendations (via sidebar radio)
- Robust data loader with error reporting and Streamlit resource caching for models and datasets
- Modular utilities: year cleaning, average rating computation, best image URL selection, and card rendering
## Data Preparation and EDA
Data is sourced from Kaggle and processed in eda.ipynb. Key steps include:
- Loading CSV files and merging books with ratings to compute average ratings per title (filtering out zero ratings and handling NaN values)
- Cleaning: Convert years to numeric, drop invalid/duplicates, and filter realistic publication years (e.g., 1900 to current year)
- Aggregations: Identify top 50 books by rating count, create a user-book pivot table for ratings, and compute cosine similarity scores
- Outputs: Save processed DataFrames and arrays as .pkl files for direct loading in app.py
Run the notebook to regenerate artifacts if needed.
## Data Artifacts Required
Place the following files in the project root (same directory as app.py)—these are generated from eda.ipynb after Kaggle data processing:
- `top_50_books.pkl`: DataFrame of the top 50 titles by rating count, used on the Top Books page
- `similarity_scores.pkl`: 2D similarity array aligned with the pivot table index
- `pt.pkl`: Pivot table of user ratings by book titles, used to resolve positions in the similarity matrix
- `books_data.pkl`: Cleaned DataFrame with columns like Book-Title, Book-Author, Publisher, Year-Of-Publication, Image-URL-* fields, and computed Avg-Rating
## How Recommendations Work
- Select a title from the pivot table index; the app locates its index and extracts the similarity row
- The list is sorted by descending similarity and the top 5 distinct titles (excluding the selected one) are returned as recommendations
- Defensive checks handle missing titles, shape mismatches, and unexpected data conditions to avoid crashes
## Requirements
Create `requirements.txt` with these packages for Community Cloud:

streamlit

pandas

numpy

requests

joblib

Pillow

scikit-learn # Added for similarity computations in eda.ipynb
