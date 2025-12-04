from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import process, fuzz
import numpy as np
import pandas as pd
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import requests
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# File paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
MOVIES_CSV = os.path.join(DATA_DIR, "clean_parsed_tmdb_5000.csv")
ORIGINAL_MOVIES_CSV = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
LINKS_CSV = os.path.join(DATA_DIR, "links.csv")

# TMDB Image Base URL
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_API_KEY = os.getenv("TMDB_API_KEY", None)

if TMDB_API_KEY:
    # Check if API key looks like a JWT token (common mistake)
    if TMDB_API_KEY.startswith("eyJ"):
        print("⚠ WARNING: API key starts with 'eyJ' - this looks like a JWT token, not a TMDB API key!")
        print("   TMDB API keys should be alphanumeric strings (e.g., 'abc123def456...')")
        print("   Get your API key from: https://www.themoviedb.org/settings/api")
        print("   Make sure you're copying the 'API Key' not an access token or session token")
    else:
        print(f"TMDB API key loaded (starts with: {TMDB_API_KEY[:8]}...)")
else:
    print("Warning: TMDB_API_KEY not set. Will only use poster_path from CSV.")

app = FastAPI(title="RottenCabbages API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
try:
    movies = pd.read_csv(MOVIES_CSV)
    movies = movies.dropna(subset=["id","title"])
    
    # Ensure required columns exist
    if "vote_average" not in movies.columns or "vote_count" not in movies.columns:
        raise ValueError("Required columns 'vote_average' or 'vote_count' not found in CSV")
    
    # Load original movies CSV to get poster_path
    try:
        original_movies = pd.read_csv(ORIGINAL_MOVIES_CSV)
        print(f"Original CSV columns: {list(original_movies.columns)}")
        
        if "poster_path" in original_movies.columns:
            original_movies = original_movies[["id", "poster_path"]].copy()
            # Filter out null poster_paths before merge to see what we have
            valid_posters = original_movies["poster_path"].notna() & (original_movies["poster_path"] != "")
            print(f"Original CSV has {valid_posters.sum()}/{len(original_movies)} movies with poster_path")
            
            # Merge poster_path into movies dataframe
            movies = movies.merge(original_movies, on="id", how="left")
            
            # Check how many posters we have after merge
            posters_count = movies["poster_path"].notna().sum()
            valid_posters_count = (movies["poster_path"].notna() & (movies["poster_path"] != "")).sum()
            total_count = len(movies)
            print(f"After merge: {valid_posters_count}/{total_count} movies with valid poster_path")
        else:
            print("Warning: poster_path column not found in original CSV")
            movies["poster_path"] = None
    except Exception as e:
        print(f"Warning: Could not load poster_path from original CSV: {e}")
        import traceback
        traceback.print_exc()
        movies["poster_path"] = None
    
    links = pd.read_csv(LINKS_CSV)[["movieId","tmdbId"]].dropna()
    links["tmdbId"] = links["tmdbId"].astype(int)
    
    title_index = movies.merge(links, left_on="id", right_on="tmdbId", how="inner")[["title","id","movieId"]]
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Cache for TMDB API calls to avoid repeated requests
_poster_cache = {}

def fetch_poster_from_tmdb(tmdb_id):
    """Fetch poster_path from TMDB API if we have an API key"""
    # Check cache first
    if tmdb_id in _poster_cache:
        return _poster_cache[tmdb_id]
    
    if not TMDB_API_KEY:
        print(f"No TMDB API key available for movie {tmdb_id}")
        return None
    
    try:
        url = f"{TMDB_API_BASE}/movie/{tmdb_id}"
        response = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                print(f"✓ Fetched poster for movie {tmdb_id}: {poster_path[:30]}...")
            else:
                print(f"⚠ Movie {tmdb_id} has no poster_path in TMDB response")
            # Cache the result (even if None)
            _poster_cache[tmdb_id] = poster_path
            return poster_path
        elif response.status_code == 401:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get("status_message", response.text[:200])
            print(f"✗ Authentication failed for movie {tmdb_id}")
            print(f"  Error: {error_msg}")
            print(f"  API Key format check: starts with 'eyJ' (JWT-like) - this might be wrong!")
            print(f"  TMDB API keys should be alphanumeric strings, not JWT tokens")
            print(f"  Get your API key from: https://www.themoviedb.org/settings/api")
        elif response.status_code == 404:
            print(f"⚠ Movie {tmdb_id} not found in TMDB")
        elif response.status_code == 429:
            print(f"⚠ Rate limited by TMDB API for movie {tmdb_id}")
        else:
            print(f"✗ Error {response.status_code} fetching poster for movie {tmdb_id}: {response.text[:100]}")
    except requests.exceptions.Timeout:
        print(f"✗ Timeout fetching poster for movie {tmdb_id}")
    except Exception as e:
        print(f"✗ Error fetching poster from TMDB for {tmdb_id}: {type(e).__name__}: {e}")
    
    # Cache None to avoid repeated failed requests
    _poster_cache[tmdb_id] = None
    return None

# Helper function to get poster URL
def get_poster_url(poster_path, tmdb_id=None):
    # First try to use poster_path from CSV
    if pd.notna(poster_path) and poster_path is not None:
        poster_path = str(poster_path).strip()
        if poster_path and poster_path != "nan" and poster_path != "None":
            # Ensure poster_path starts with /
            if not poster_path.startswith("/"):
                poster_path = "/" + poster_path
            return f"{TMDB_IMAGE_BASE}{poster_path}"
    
    # If no poster_path and we have API key, try fetching from TMDB
    if tmdb_id and TMDB_API_KEY:
        fetched_path = fetch_poster_from_tmdb(tmdb_id)
        if fetched_path:
            if not fetched_path.startswith("/"):
                fetched_path = "/" + fetched_path
            return f"{TMDB_IMAGE_BASE}{fetched_path}"
    
    return None

######################
# DEMOGRAPHIC BASED TRENDING MOVIES
def compute_trending_movies(top_percent = 30, top_k = 20):
    # Mean average rating across all movies
    mean_rating = movies["vote_average"].mean()
    
    # Handle case where mean_rating might be NaN
    if pd.isna(mean_rating):
        mean_rating = 0.0
    
    quant = 1 - (top_percent / 100)
    # Minimum votes required to be in the top %
    minimum_votes = movies['vote_count'].quantile(quant)
    
    # Handle case where minimum_votes might be NaN
    if pd.isna(minimum_votes):
        minimum_votes = 0.0
    
    topMovies = movies.copy().loc[movies['vote_count'] >= minimum_votes]
    
    # If no movies meet the criteria, return empty list
    if len(topMovies) == 0:
        return []
    
    def weighted_rating(x, m=minimum_votes, C=mean_rating):
        v = x['vote_count']
        R = x['vote_average']
        # IMDB formula for weighted rating
        return (v / (v + m) * R) + (m / (m + v) * C)

    # Calculate weighted score
    topMovies['score'] = topMovies.apply(weighted_rating, axis=1) 
    
    # Sort movies by score
    topMovies = topMovies.sort_values('score', ascending=False)
    
    # Convert to list of dictionaries and fetch missing posters
    result = []
    for r in topMovies.head(top_k).to_dict(orient="records"):
        genres_str = str(r.get("genres",""))
        genres_list = [g.strip() for g in genres_str.split(",") if g.strip()] if genres_str else []
        poster_path = r.get("poster_path")
        tmdb_id = int(r["id"])
        
        # Ensure we have a poster URL (fetch from API if missing)
        poster_url = get_poster_url(poster_path, tmdb_id)
        
        result.append({
            "tmdb_id": tmdb_id,
            "title": str(r["title"]),
            "genres": genres_list,
            "rating": float(r.get("vote_average", 0)),
            "vote_average": float(r.get("vote_average", 0)),
            "vote_count": int(r.get("vote_count", 0)),
            "score": float(r.get("score", 0)),
            "poster_url": poster_url
        })
    
    # Log how many posters we have
    posters_count = sum(1 for item in result if item["poster_url"])
    print(f"Trending movies: {posters_count}/{len(result)} have posters")
    
    return result

trendingMovies = compute_trending_movies(top_percent = 30, top_k = 20)

# Pre-fetch posters for trending movies in the background
if TMDB_API_KEY:
    print(f"\n{'='*60}")
    print("Pre-fetching posters for trending movies...")
    print(f"API Key present: Yes (starts with: {TMDB_API_KEY[:8]}...)")
    print(f"Trending movies to fetch: {len(trendingMovies)}")
    print(f"{'='*60}\n")
    
    for i, movie in enumerate(trendingMovies, 1):
        tmdb_id = movie.get("tmdb_id")
        title = movie.get("title", "Unknown")
        print(f"[{i}/{len(trendingMovies)}] Fetching poster for: {title} (ID: {tmdb_id})")
        
        if tmdb_id and not movie.get("poster_url"):
            # Fetch poster if missing
            poster_path = fetch_poster_from_tmdb(tmdb_id)
            if poster_path:
                if not poster_path.startswith("/"):
                    poster_path = "/" + poster_path
                movie["poster_url"] = f"{TMDB_IMAGE_BASE}{poster_path}"
    
    posters_fetched = sum(1 for m in trendingMovies if m.get("poster_url"))
    print(f"\n{'='*60}")
    print(f"Pre-fetch complete: {posters_fetched}/{len(trendingMovies)} trending movies now have posters")
    print(f"{'='*60}\n")
else:
    print("⚠ TMDB_API_KEY not set - cannot fetch posters from API")
###################

# Prepare movie titles for fuzzy search (done once at startup)
movie_titles_list = movies["title"].astype(str).tolist()
print(f"Prepared {len(movie_titles_list)} movie titles for fuzzy search")

# Build content-based similarity matrix (from content_based_filtering notebook)
print("Building content-based similarity matrix...")
try:
    # Combine all content into a single text field with weighted importance
    # Genres are repeated 4x to give them more weight in similarity calculations
    # Production companies are repeated 2x to give them more weight
    movies["combined_content"] = (
        movies["genres"].astype(str) + " " +
        movies["genres"].astype(str) + " " +  # Repeat genres for extra weight
        movies["genres"].astype(str) + " " +  # Repeat genres again
        movies["genres"].astype(str) + " " +  # Repeat genres one more time (4x total)
        movies["keywords"].astype(str) + " " +
        movies["overview"].astype(str) + " " +
        movies["production_companies"].astype(str) + " " +
        movies["production_companies"].astype(str) + " " +  # Repeat production companies for 2x weight
        movies["tagline"].astype(str) + " " +
        movies["cast"].astype(str)
    )
    
    # Convert to lowercase and clean
    movies["combined_content"] = movies["combined_content"].astype(str).str.lower()
    movies["combined_content"] = movies["combined_content"].str.replace(r'[^\w\s]', '', regex=True)
    movies["combined_content"] = movies["combined_content"].str.strip()
    
    # Create TF-IDF + SVD pipeline (same as notebook)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    svd = TruncatedSVD(n_components=150, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa_pipeline = make_pipeline(vectorizer, svd, normalizer)
    
    # Create movie embeddings
    movie_embeddings = lsa_pipeline.fit_transform(movies["combined_content"])
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(movie_embeddings)
    
    print(f"✓ Content-based similarity matrix built: {similarity_matrix.shape}")
    print(f"  Using TF-IDF + SVD (150 components) + Cosine Similarity")
    print(f"  Genres weighted 4x, Production companies weighted 2x")
except Exception as e:
    print(f"⚠ Error building similarity matrix: {e}")
    import traceback
    traceback.print_exc()
    similarity_matrix = None

###################
# ENDPOINTS
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/trending")
def trending(limit: int = 12):
    return trendingMovies[:limit]

@app.get("/movie/{tmdb_id}")
def get_movie(tmdb_id: int):
    # Find movie by tmdb id
    movie = movies[movies["id"] == tmdb_id]
    
    if movie.empty:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Movie not found")
    
    m = movie.iloc[0].to_dict()
    
    # Parse genres (comma-separated string)
    genres_str = str(m.get("genres", ""))
    genres_list = [g.strip() for g in genres_str.split(",") if g.strip()] if genres_str else []
    
    # Parse cast (comma-separated string)
    cast_str = str(m.get("cast", ""))
    cast_list = [c.strip() for c in cast_str.split(",") if c.strip()] if cast_str else []
    
    # Parse crew (comma-separated string with job descriptions)
    crew_str = str(m.get("crew", ""))
    crew_list = [c.strip() for c in crew_str.split(",") if c.strip()] if crew_str else []
    
    # Parse production companies
    prod_companies_str = str(m.get("production_companies", ""))
    prod_companies = [c.strip() for c in prod_companies_str.split(",") if c.strip()] if prod_companies_str else []
    
    # Parse production countries
    prod_countries_str = str(m.get("production_countries", ""))
    prod_countries = [c.strip() for c in prod_countries_str.split(",") if c.strip()] if prod_countries_str else []
    
    # Format release date
    release_date = m.get("release_date")
    if pd.notna(release_date):
        if isinstance(release_date, str):
            release_date = release_date
        else:
            release_date = str(release_date)
    else:
        release_date = None
    
    # Get poster URL - always try to fetch if missing
    poster_path = m.get("poster_path")
    tmdb_id = int(m["id"])
    poster_url = get_poster_url(poster_path, tmdb_id)
    
    if not poster_url and TMDB_API_KEY:
        # Try fetching from API one more time
        fetched_path = fetch_poster_from_tmdb(tmdb_id)
        if fetched_path:
            if not fetched_path.startswith("/"):
                fetched_path = "/" + fetched_path
            poster_url = f"{TMDB_IMAGE_BASE}{fetched_path}"
    
    return {
        "tmdb_id": int(m["id"]),
        "title": str(m.get("title", "")),
        "overview": str(m.get("overview", "")),
        "genres": genres_list,
        "cast": cast_list,
        "crew": crew_list,
        "rating": float(m.get("vote_average", 0)),
        "vote_count": int(m.get("vote_count", 0)),
        "release_date": release_date,
        "runtime": int(m.get("runtime", 0)) if pd.notna(m.get("runtime")) else None,
        "tagline": str(m.get("tagline", "")) if pd.notna(m.get("tagline")) else None,
        "production_companies": prod_companies,
        "production_countries": prod_countries,
        "poster_url": poster_url
    }

@app.get("/suggest")
def suggest_movies(query: str, limit: int = 10):
    """
    Search for movies with improved relevance ranking.
    Prioritizes exact/prefix matches, then word matches, then fuzzy matches.
    """
    if not query or len(query.strip()) < 2:
        return []
    
    query = query.strip().lower()
    query_words = query.split()
    
    # Categorize matches by relevance
    exact_start_matches = []  # Titles that start with query
    word_matches = []  # Titles that contain query as a word
    fuzzy_matches = []  # Fuzzy matches with higher threshold
    
    for idx, title in enumerate(movie_titles_list):
        title_lower = str(title).lower()
        
        # Priority 1: Exact start match (case-insensitive)
        if title_lower.startswith(query):
            exact_start_matches.append((title, 100, idx))
        # Priority 2: Word boundary match (query appears as a complete word)
        else:
            # Check if query appears as a whole word
            word_pattern = r'\b' + re.escape(query) + r'\b'
            if re.search(word_pattern, title_lower):
                word_matches.append((title, 90, idx))
            # Check if individual query words appear as whole words
            elif len(query_words) > 1:
                matching_words = sum(1 for word in query_words if len(word) >= 3 and re.search(r'\b' + re.escape(word) + r'\b', title_lower))
                if matching_words > 0:
                    word_matches.append((title, 80 + matching_words * 5, idx))
            # Priority 3: Fuzzy match with stricter threshold
            else:
                # Use token_sort_ratio which is better for word order independence
                # but still requires the words to be present (stricter than WRatio)
                score = fuzz.token_sort_ratio(query, title_lower)
                # Also check that query isn't just embedded in a longer word
                # (e.g., "fight" shouldn't match "Foodfight" well)
                if score >= 80:  # Higher threshold for stricter matching
                    # Additional check: if query is short, require it to be a word boundary
                    if len(query) <= 5:
                        # For short queries, be extra strict - require word boundary
                        if not re.search(r'\b' + re.escape(query) + r'\b', title_lower):
                            # If no word boundary match, use even higher threshold
                            if score < 90:
                                continue
                    fuzzy_matches.append((title, score, idx))
    
    # Combine and sort: exact start > word match > fuzzy match
    all_matches = exact_start_matches + word_matches + sorted(fuzzy_matches, key=lambda x: x[1], reverse=True)
    
    # Remove duplicates (keep first occurrence)
    seen_titles = set()
    unique_matches = []
    for title, score, idx in all_matches:
        if title not in seen_titles:
            seen_titles.add(title)
            unique_matches.append((title, score, idx))
    
    # Limit results
    filtered_matches = unique_matches[:limit]
    
    # Build response with poster URLs
    result = []
    for title, score, _ in filtered_matches:
        # Find movie by title
        movie = movies[movies["title"] == title]
        if not movie.empty:
            m = movie.iloc[0].to_dict()
            genres_str = str(m.get("genres", ""))
            genres_list = [g.strip() for g in genres_str.split(",") if g.strip()] if genres_str else []
            poster_path = m.get("poster_path")
            tmdb_id = int(m["id"])
            
            # Get poster URL (fetch from API if missing)
            poster_url = get_poster_url(poster_path, tmdb_id)
            
            result.append({
                "tmdb_id": tmdb_id,
                "title": str(title),
                "genres": genres_list,
                "rating": float(m.get("vote_average", 0)),
                "poster_url": poster_url,
                "score": score  # Similarity score for debugging
            })
    
    return result

@app.get("/movie/{tmdb_id}/users_also_watched")
def get_users_also_watched(tmdb_id: int, k: int = 10):
    """
    Get movies that users also watched using content-based filtering.
    Uses TF-IDF + SVD + Cosine Similarity (same as content_based_filtering notebook).
    """
    # Find the current movie
    current_movie = movies[movies["id"] == tmdb_id]
    
    if current_movie.empty:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Movie not found")
    
    # Use content-based similarity matrix if available
    if similarity_matrix is not None:
        # Get movie index
        movie_idx = current_movie.index[0]
        
        # Get similarity scores for this movie
        scores = similarity_matrix[movie_idx]
        
        # Find top k similar movies (excluding the movie itself)
        similar_indices = scores.argsort()[::-1][1:k+1]  # Skip index 0 (itself)
        
        # Get similar movies
        similar_movies = movies.iloc[similar_indices].copy()
        similar_scores = scores[similar_indices]
    else:
        # Fallback to simple genre-based matching if similarity matrix not available
        print("⚠ Using fallback genre-based matching (similarity matrix not available)")
        current_genres = str(current_movie.iloc[0].get("genres", "")).lower()
        similar_movies = movies[movies["id"] != tmdb_id].copy()
        
        def similarity_score(row):
            score = 0
            row_genres = str(row.get("genres", "")).lower()
            if current_genres and row_genres:
                current_genre_list = [g.strip() for g in current_genres.split(",") if g.strip()]
                row_genre_list = [g.strip() for g in row_genres.split(",") if g.strip()]
                common_genres = set(current_genre_list) & set(row_genre_list)
                score += len(common_genres) * 2
            vote_avg = row.get("vote_average", 0)
            if pd.notna(vote_avg):
                score += float(vote_avg) / 10
            return score
        
        similar_movies["similarity_score"] = similar_movies.apply(similarity_score, axis=1)
        similar_movies = similar_movies.sort_values("similarity_score", ascending=False).head(k)
        similar_scores = similar_movies["similarity_score"].values
    
    # Get top k similar movies
    result = []
    for idx, (_, r) in enumerate(similar_movies.iterrows()):
        m = r.to_dict()
        genres_str = str(m.get("genres", ""))
        genres_list = [g.strip() for g in genres_str.split(",") if g.strip()] if genres_str else []
        poster_path = m.get("poster_path")
        movie_tmdb_id = int(m["id"])
        
        # Always try to get poster URL (fetch from API if missing)
        poster_url = get_poster_url(poster_path, movie_tmdb_id)
        
        result.append({
            "tmdb_id": movie_tmdb_id,
            "title": str(m.get("title", "")),
            "genres": genres_list,
            "rating": float(m.get("vote_average", 0)),
            "poster_url": poster_url
        })
    
    # Log poster status
    posters_count = sum(1 for item in result if item["poster_url"])
    print(f"Users also watched: {posters_count}/{len(result)} movies have posters")
    
    return result