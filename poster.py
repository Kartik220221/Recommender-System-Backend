from dotenv import load_dotenv
import httpx
import os 
import asyncio

load_dotenv()

api_key = os.getenv("OMDB_API_KEY")


if not api_key:
    print("no api key found")


async def get_omdb_poster(titles:list):
    poster_list = []
    for i in titles:
        url = 'http://www.omdbapi.com'
        params={
            "t":i,
            'apiKey':api_key
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url,params=params)
            if response.status_code==200:
                data = response.json()
                if(data.get("Response")=="True"):
                    poster_list.append(data.get("Poster"))
                else:
                    poster_list.append('https://tse2.mm.bing.net/th?id=OIP.Skr-oJ6BWg_K65k5uDiMdgHaHa&pid=Api&P=0&h=180')
            else:
                poster_list.append('https://tse2.mm.bing.net/th?id=OIP.Skr-oJ6BWg_K65k5uDiMdgHaHa&pid=Api&P=0&h=180')
    return poster_list



if __name__=="__main__":
    movie_titles = [
    "The Shawshank Redemption",
    "The Dark Knight",
    "Inception",
    "Forrest Gump",
    "The Matrix"
    ]
    poster_urls = asyncio.run(get_omdb_poster(titles=movie_titles))
    print(poster_urls)