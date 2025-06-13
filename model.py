from pydantic import BaseModel
from typing import Dict,List,Optional

class MovieModel(BaseModel):
    movie_title:str

class MovieOutputModel(BaseModel):
    movie_titles:List[str]
    movie_images:List[str]
    