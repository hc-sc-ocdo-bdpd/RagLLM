# Code adapted from https://github.com/Azure/azureml-examples/blob/main/sdk/python/generative-ai/promptflow/create_faiss_index.ipynb

# Import libraries
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import matplotlib.pyplot as plt 
import numpy as np 
import faiss
from openai import AzureOpenAI

MODEL_API_VERSION = "2023-05-15"
MODEL_DEPLOYMENT_NAME = "ada_embedding"
DIMENSION = 1536
M = 8
NBITS = 8

# TODO: Figure out how to incrementally add docs to index
# TODO: Save index to local storage

def get_embeddings(client: AzureOpenAI, file_name: str) -> List[int]:
    with open(file_name, "r", encoding="utf-8") as f:
        page_content = f.read()
        doc_embeddings = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=10)
        for chunk in splitter.split_text(page_content):
            response = client.embeddings.create(
                model = MODEL_DEPLOYMENT_NAME, 
                input = chunk
            )
            doc_embeddings.append(np.array(response.data[0].embedding))
    return doc_embeddings

def create_index(file_path, store_name, graph = False):
    '''
    Creates a Faiss index for a specific set of documents.
    Args:   file_path (str): path to the folder of documnts to be indexed.
            store_name (str): intended name for the file store
            graph (bool): should a graph showing the number of documents indexed versus elapsed time be produced?
    Returns: the created file store
    '''

    # Prepare data
    local_file_path = file_path

    # Configure and create an embedding store
    client = AzureOpenAI(
        azure_deployment = MODEL_DEPLOYMENT_NAME, 
        api_version = MODEL_API_VERSION,
        azure_endpoint = os.environ["OPENAI_MODEL_ENDPOINT"],
        # openai_api_type = "azure",
        api_key = os.environ["OPENAI_API_KEY"]
    )

    # Count the total number of files for the progress bar
    total_files = sum([len(files) for _, _, files in os.walk(local_file_path)])
    progress_bar = tqdm(total=total_files, desc="Processing Files")
    elapsed_times = []
    chunks = []
    metadatas = []
    embeddings = []
 
    for root, _, files in os.walk(local_file_path):
        for file in files:
            each_file_path = os.path.join(root, file)

            with open(each_file_path, "r", encoding="utf-8") as f:
                title = f.readline().strip('\n')

            # Split the file into chunks.
            embeddings.extend(get_embeddings(client, each_file_path))
            # TODO: Update count
            count = len(embeddings)
            metadatas.extend([
                {"title": title, "source": os.path.join(file)}
            ] * count)

            progress_bar.update()
            if graph:
                elapsed_times.append(progress_bar.format_dict['elapsed'])

    xb = np.array(embeddings)
    # print(xb.shape)

    index = faiss.IndexPQ(DIMENSION, M, NBITS)
    index.train(xb)

    progress_bar.close()
    # create line graph
    if graph:  
        x = np.array(range(progress_bar.format_dict['total']))
        y = np.array(elapsed_times)
        
        plt.plot(x, y) 
        plt.xlabel("# of Documents Indexed") 
        plt.ylabel("Elapsed Time (seconds)")
        plt.title("# of Documents Indexed versus Elapsed Time")
        plt.show()

    return index