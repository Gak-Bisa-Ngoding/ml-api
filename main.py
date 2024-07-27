from typing import Union, List, Dict
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

data = pd.read_csv("./data/data.csv")
cos_sim_df = pd.read_csv("./data/cos_sim_df.csv")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/recommend/{name}")
def recommend(name: str):
    recom_data = get_recom(name)
    return recom_data


def get_recom(name: str, similarity_data=cos_sim_df, items=data[['title', 'genre']], k=5) -> List[Dict]:
    index = similarity_data.loc[:, name].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(name, errors='ignore')
    
    # Creating a DataFrame from closest names
    closest_df = pd.DataFrame(closest, columns=['title'])
    
    # Merging with items DataFrame on 'title' column
    merged_df = closest_df.merge(items, on='title').head(k)
    
    # Converting to list of dictionaries
    return merged_df.to_dict(orient='records')