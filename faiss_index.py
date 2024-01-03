# Code adapted from https://github.com/Azure/azureml-examples/blob/main/sdk/python/generative-ai/promptflow/create_faiss_index.ipynb

# Import libraries
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from promptflow_vectordb.core.contracts import (
    EmbeddingModelType,
    StorageType,
    StoreCoreConfig,
)
from promptflow_vectordb.core.embeddingstore_core import EmbeddingStoreCore

def create_index(file_path, store_name):

    # Prepare data
    local_file_path = file_path

    # Configure and create an embedding store
    MODEL_API_VERSION = "2023-05-15"
    MODEL_DEPLOYMENT_NAME = "ada_embedding"
    DIMENSION = 1536
    os.environ["Azure_OpenAI_MODEL_ENDPOINT"] = "<ENTER ENDPOINT HERE>"
    os.environ["Azure_OpenAI_MODEL_API_KEY"] = "<ENTER API KEY>"

    # Configure an embedding store to store index file
    store_path = os.path.join(os.getcwd(), store_name)
    config = StoreCoreConfig.create_config(
        storage_type=StorageType.LOCAL,
        store_identifier=store_path,
        model_type=EmbeddingModelType.AOAI,
        model_api_base=os.environ["Azure_OpenAI_MODEL_ENDPOINT"],
        model_api_key=os.environ["Azure_OpenAI_MODEL_API_KEY"],
        model_api_version=MODEL_API_VERSION,
        model_name=MODEL_DEPLOYMENT_NAME,
        dimension=DIMENSION,
        create_if_not_exists=True,
    )
    store = EmbeddingStoreCore(config)

    # Split document to chunks, embed chunks and create Faiss index
    def get_file_chunks(file_name: str) -> List[str]:
        with open(file_name, "r", encoding="utf-8") as f:
            page_content = f.read()
            chunks = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=10)
            for chunk in splitter.split_text(page_content):
                chunks.append(chunk)
            return chunks
        
    # Count the total number of files for the progress bar
    total_files = sum([len(files) for _, _, files in os.walk(local_file_path)])
    progress_bar = tqdm(total=total_files, desc="Processing Files")
 
    for root, _, files in os.walk(local_file_path):
        for file in files:
            each_file_path = os.path.join(root, file)

            with open(each_file_path, "r", encoding="utf-8") as f:
                title = f.readline().strip('\n')

            # Split the file into chunks.
            chunks = get_file_chunks(each_file_path)
            count = len(chunks)
            metadatas = [
                {"title": title, "source": os.path.join(file)}
            ] * count

            # Embed chunks into embeddings, generate index in embedding store.
            # If your data is large, inserting too many chunks at once may cause
            # rate limit error，you can refer to the following link to find solution
            # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quotas-limits
            store.batch_insert_texts(chunks, metadatas)
            progress_bar.update()

    progress_bar.close() 
    return store