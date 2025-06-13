import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')
except:
    print("NLTK downloads completed")

# Load your datasets
# Replace with your actual file paths
df = pd.read_csv('movies_dataset.csv')  # Your main movies dataset
credits = pd.read_csv('credits.csv')    # Your credits dataset

print("Initial data loaded:")
print(f"Movies dataset shape: {df.shape}")
print(f"Credits dataset shape: {credits.shape}")

# ===== DATA PREPROCESSING =====

def convert_to_list(text):
    """Convert string representation of list to actual list"""
    try:
        return ast.literal_eval(text)
    except:
        return []

def get_list_of_names(obj_list):
    """Extract names from list of dictionaries"""
    if isinstance(obj_list, list):
        return [obj['name'] for obj in obj_list if isinstance(obj, dict) and 'name' in obj]
    return []

def get_director(crew_list):
    """Extract director from crew list"""
    if isinstance(crew_list, list):
        for person in crew_list:
            if isinstance(person, dict) and person.get('job') == 'Director':
                return person.get('name', '')
    return ''

# Process the main dataset
print("\nProcessing main dataset...")

# Convert string representations to lists
list_columns = ['genres', 'keywords', 'production_countries']
for col in list_columns:
    if col in df.columns:
        df[col] = df[col].apply(convert_to_list)
        df[col] = df[col].apply(get_list_of_names)

# Process credits dataset
print("Processing credits dataset...")
if 'cast' in credits.columns:
    credits['cast'] = credits['cast'].apply(convert_to_list)
    credits['cast'] = credits['cast'].apply(lambda x: [obj['name'] for obj in x[:5]] if isinstance(x, list) else [])

if 'crew' in credits.columns:
    credits['crew'] = credits['crew'].apply(convert_to_list)
    credits['director'] = credits['crew'].apply(get_director)

# Merge datasets
print("Merging datasets...")
df = df.merge(credits, on='title', how='left')

# Drop unnecessary columns
print("Dropping unnecessary columns...")
columns_to_drop = ['homepage','movie_id','budget','original_language','original_title',
                   'popularity','production_companies','production_countries',
                   'release_date','revenue','runtime','spoken_languages',
                   'status','tagline','vote_average','vote_count', 'crew']

# Only drop columns that actually exist
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
if existing_columns_to_drop:
    df.drop(existing_columns_to_drop, axis=1, inplace=True)

print(f"Dataset shape after merging and dropping: {df.shape}")

# ===== TEXT PROCESSING FUNCTIONS =====

def merge_words(list_of_strings):
    """Remove spaces from list of strings"""
    if isinstance(list_of_strings, list):
        return [s.replace(' ', '') if isinstance(s, str) else s for s in list_of_strings]
    return list_of_strings

def convert_overview(text):
    """Convert overview text to list of words"""
    if pd.isna(text) or text == '':
        return []
    return text.split(' ')

# Apply text processing
print("Processing text data...")

# Remove spaces from list columns
list_cols = ['genres', 'keywords', 'cast']
for col in list_cols:
    if col in df.columns:
        df[col] = df[col].apply(merge_words)

# Convert overview to word list
if 'overview' in df.columns:
    df['overview'] = df['overview'].apply(convert_overview)

# Add director to list format
if 'director' in df.columns:
    df['director'] = df['director'].apply(lambda x: [x] if isinstance(x, str) and x != '' else [])

# Create final text by combining all features
print("Creating final text features...")

def create_final_text(row):
    """Combine all text features into single list"""
    final_text = []
    
    # Add each feature if it exists and is a list
    features = ['genres', 'keywords', 'overview', 'cast', 'director']
    for feature in features:
        if feature in row and isinstance(row[feature], list):
            final_text.extend(row[feature])
    
    return final_text

df['final_text'] = df.apply(create_final_text, axis=1)

# Reset index to ensure continuous indexing
df = df.reset_index(drop=True)

print(f"Final dataset shape: {df.shape}")
print("Sample of final_text:")
if len(df) > 0:
    print(df['final_text'][0][:10])  # Show first 10 words

# ===== NLTK PROCESSING =====

print("\nStarting NLTK processing...")

# Initialize NLTK tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(word_list):
    """Process a list of words: remove stopwords, convert to lowercase, and lemmatize"""
    if not isinstance(word_list, list):
        return []
    
    processed_words = []
    for word in word_list:
        if isinstance(word, str):
            # Convert to lowercase and remove punctuation
            word = word.lower().strip('.,!?";:()[]{}')
            
            # Skip if it's a stopword or empty
            if word not in stop_words and word != '' and len(word) > 1:
                # Lemmatize (generally better than stemming for movie recommendations)
                try:
                    lemmatized_word = lemmatizer.lemmatize(word)
                    processed_words.append(lemmatized_word)
                except:
                    processed_words.append(word)
    
    return processed_words

def create_text_from_processed_words(word_list):
    """Convert processed word list back to a single string for vectorization"""
    if isinstance(word_list, list):
        return ' '.join(word_list)
    return ''

# Apply preprocessing
print("Step 1: Preprocessing text (removing stopwords, lemmatizing)")
df['processed_text'] = df['final_text'].apply(preprocess_text)

# Convert back to strings for vectorization
print("Step 2: Converting processed words back to text strings")
df['clean_text'] = df['processed_text'].apply(create_text_from_processed_words)

# Remove rows with empty clean_text
df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
print(f"Dataset shape after removing empty text: {df.shape}")

# ===== VECTORIZATION =====

print("\nStep 3: Creating TF-IDF vectors")

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit to most common 5000 words
    ngram_range=(1, 2),  # Include both single words and bigrams
    min_df=2,  # Ignore words that appear in less than 2 documents
    max_df=0.8  # Ignore words that appear in more than 80% of documents
)

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")

# Calculate cosine similarity matrix
print("Step 4: Calculating cosine similarity matrix")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"Cosine similarity matrix shape: {cosine_sim.shape}")

# ===== RECOMMENDATION FUNCTION =====

def get_movie_recommendations(movie_title, df, cosine_sim_matrix, top_n=10):
    """Get movie recommendations based on cosine similarity"""
    
    # Debug information
    print(f"DataFrame shape: {df.shape}")
    print(f"Cosine similarity matrix shape: {cosine_sim_matrix.shape}")
    
    # Get the index of the movie
    movie_matches = df[df['title'].str.lower() == movie_title.lower()]
    
    if movie_matches.empty:
        print(f"Movie '{movie_title}' not found in database")
        print("Available movies (first 10):")
        print(df['title'].head(10).tolist())
        return None
    
    movie_idx = movie_matches.index[0]
    print(f"Found movie '{movie_title}' at index: {movie_idx}")
    
    # Validate that the index is within bounds
    if movie_idx >= cosine_sim_matrix.shape[0]:
        print(f"Error: Movie index {movie_idx} is out of bounds for similarity matrix")
        return None
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim_matrix[movie_idx]))
    
    # Sort movies by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return movie titles and similarity scores
    recommendations = df.iloc[movie_indices][['title', 'id']].copy()
    recommendations['similarity_score'] = [score[1] for score in sim_scores]
    
    return recommendations

def search_movies(query, df, top_n=10):
    """Search for movies by title"""
    matches = df[df['title'].str.lower().str.contains(query.lower(), na=False)]
    return matches[['title', 'id']].head(top_n)

# ===== TESTING THE SYSTEM =====

print("\n" + "="*50)
print("MOVIE RECOMMENDATION SYSTEM READY!")
print("="*50)

# Test the system
print("\nTesting the recommendation system...")

# Show sample movies
print("\nSample movies in database:")
print(df[['title', 'id']].head(10))

# Test with a specific movie
test_movie = df['title'].iloc[0] if len(df) > 0 else 'Avatar'
print(f"\nGetting recommendations for '{test_movie}':")

try:
    recommendations = get_movie_recommendations(test_movie, df, cosine_sim, top_n=5)
    if recommendations is not None:
        print(recommendations)
    else:
        print("No recommendations generated")
except Exception as e:
    print(f"Error getting recommendations: {e}")

# ===== UTILITY FUNCTIONS =====

def save_model(df, cosine_sim, tfidf_matrix, filename_prefix='movie_rec'):
    """Save the processed data and model"""
    df.to_csv(f'{filename_prefix}_processed.csv', index=False)
    np.save(f'{filename_prefix}_cosine_sim.npy', cosine_sim)
    np.save(f'{filename_prefix}_tfidf.npy', tfidf_matrix.toarray())
    print(f"Model saved with prefix: {filename_prefix}")

def load_model(filename_prefix='movie_rec'):
    """Load the processed data and model"""
    df = pd.read_csv(f'{filename_prefix}_processed.csv')
    cosine_sim = np.load(f'{filename_prefix}_cosine_sim.npy')
    tfidf_matrix = np.load(f'{filename_prefix}_tfidf.npy')
    return df, cosine_sim, tfidf_matrix

# Save the model for future use
print("\nSaving model for future use...")
try:
    save_model(df, cosine_sim, tfidf_matrix)
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")

print("\n" + "="*50)
print("SETUP COMPLETE!")
print("="*50)
print("\nYou can now use:")
print("1. get_movie_recommendations('Movie Title', df, cosine_sim, top_n=10)")
print("2. search_movies('search term', df)")
print("3. save_model() and load_model() for persistence")