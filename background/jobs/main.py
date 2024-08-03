import pandas as pd
from sentence_transformers import SentenceTransformer
from background.database import Database
# import os
def main():
    file = 'company1.xlsx'
    # print("Current Working Directory:", os.getcwd())
    df = pd.read_excel(file, engine='openpyxl')
    df = df.drop(df.columns[0], axis=1)
    titles = df['Title '].tolist()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    embeddings = model.encode(titles)
    df['embedding'] = [embedding.tolist() for embedding in embeddings]
    records = df.to_dict(orient='records')

    db = Database(file)
    db.save(records)

if __name__ == "__main__":
    main()