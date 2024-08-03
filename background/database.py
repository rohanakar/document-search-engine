from typing import List
from pymilvus import MilvusClient, DataType
from pymilvus import utility

class Database:

    db_file = 'database.db'
    
    def __init__(self, data_file:str):
        self.data_file = data_file
        self.collection_name = self.data_file.split(".")[0]
        self.client = MilvusClient(Database.db_file)
        self.create_collection()

    def create_collection(self):
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )

        # 2. Add fields to schema
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, is_primary=False)
        schema.add_field(field_name="summary", datatype=DataType.VARCHAR, is_primary=False)
        schema.add_field(field_name="vector_embedding", datatype=DataType.FLOAT_VECTOR, dim=384 )

        # 3. Prepare index parameters
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="vector-embedding", 
            index_type="AUTOINDEX",
            metric_type="COSINE",
            params={"nlist": 384 }
        )

        self.client.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )



    def save(self,reports:List[dict]):

        self.client.insert(
            collection_name=self.collection_name,
            data=[reports]
        )

