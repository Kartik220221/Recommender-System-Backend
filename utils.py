import pandas as pd
import numpy as np
import requests
import os

def download_if_not_exists(url,local_path):
    if not os.path.exists(local_path):
        print("downloading local path")
        response = requests.get(url)
        with open(local_path,'wb') as f:
            f.write(response.content)


def load_model(filename_prefix='rec_files/movie_rec'):
    os.makedirs('rec_files', exist_ok=True)

    csv_url = 'https://drive.google.com/file/d/18f1Ocfr4qXhS8JxxBokdCU2V9bcC8C3J/view?usp=sharing'
    cosine_url = 'https://drive.google.com/file/d/1uxKdfyUaPflp02A0ef8EFHdaaIV6Rt-a/view?usp=sharing'
    tfidf_url = 'https://drive.google.com/file/d/1vOtvz_3gNNtH9ykMFqULsOZezwkx8-2z/view?usp=sharing'

    csv_path = f'{filename_prefix}_processed.csv'
    cosine_path = f"{filename_prefix}_cosine_sim.npy"
    tfidf_path = f"{filename_prefix}_tfidf.npy"

    download_if_not_exists(csv_url, csv_path)
    download_if_not_exists(cosine_url, cosine_path)
    download_if_not_exists(tfidf_url, tfidf_path)
    df = pd.read_csv(csv_path)
    cosine_np = np.load(cosine_path)
    tfidf_matrix = np.load(tfidf_path)
    return df,cosine_np,tfidf_matrix

def get_movie_recommendations(movie_title,df,cosine_matrix,top_n=5):
    print(movie_title.lower())
    movie_index = df[df['title'].str.lower()==movie_title.lower()].index[0]
    print(movie_index)
    print("Titles in dataset (first 10):")
    print(df['title'].head(10).tolist())

    print("\nAll lowercase titles:")
    print(df['title'].str.lower().tolist()[:10])
    if movie_index<0:
        print(f"{movie_title} does not exist in the dataset")
        return f"{movie_title} does not exist in the dataset"
    if movie_index>=cosine_matrix.shape[0]:
        print(f"{movie_title} is out of bounds at {movie_index}")
        return f"{movie_title} is out of bounds at {movie_index}"

    sim_scores = list(enumerate(cosine_matrix[movie_index]))
    sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]

    recommendations = df.iloc[movie_indices][['title','id']].copy()
    recommendations['similarity_score'] = [score[1] for score in sim_scores]

    return recommendations

def search_movies(query,df,top_n=10):
    search_matches = df[df['title'].str.lower().str.contains(query.lower(),na=False)]
    return search_matches[['title','id']].head(top_n)

if __name__=="__main__":
    print("loading the saved model")
    df,cosine_matrix,tfidf_matrix = load_model()
    movie_name = "Avatar"
    recommendations = get_movie_recommendations(movie_title=movie_name,df=df,cosine_matrix=cosine_matrix)
    print(recommendations['title'].tolist())



