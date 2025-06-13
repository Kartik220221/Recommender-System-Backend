from fastapi import FastAPI,APIRouter,Request,Response
from typing import List,Dict,Optional
from bson import ObjectId
from utils import *
from model import MovieModel,MovieOutputModel
from poster import get_omdb_poster
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:5173", 
    "http://localhost:3000",  
    "https://your-production-frontend.com"  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             
    allow_credentials=True,
    allow_methods=["*"],             
    allow_headers=["*"],     
)

router = APIRouter(prefix='/movie')

@router.post('/recommend',response_model=MovieOutputModel)
async def recommend_movie(movie_title:MovieModel):
    movie_title_string = movie_title.model_dump().get("movie_title")
    df,cosine_matrix,tfidf_matrix = load_model()
    recommendations_df = get_movie_recommendations(movie_title_string,df,cosine_matrix,top_n=6)
    recommendations_list = recommendations_df['title'].to_list()
    image_list = await get_omdb_poster(recommendations_list)
    output = {'movie_titles':recommendations_list,"movie_images":image_list}
    return output

app.include_router(router)



