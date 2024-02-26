# Import libraries
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from tqdm import tqdm
import matplotlib.pyplot as plt 
import numpy as np 
import faiss
from openai import AzureOpenAI
from pathlib import Path
import pickle
import uuid
from tenacity import retry, stop_after_attempt, wait_random_exponential

MODEL_API_VERSION = "2023-05-15"
MODEL_DEPLOYMENT_NAME = "ada_embedding"
DIMENSION = 1536
NLIST = 64
MAX_ARRAY_SIZE = 2048

# Create chunks and embeddings
def get_chunks(file_name: str):
    with open(file_name, "r", encoding="utf-8") as f:
        page_content = f.read()
        chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=10)
        for chunk in splitter.split_text(page_content):
            chunks.append(chunk)
    return chunks

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def create_embeddings(client: AzureOpenAI, chunks):
    response = client.embeddings.create(
        model = MODEL_DEPLOYMENT_NAME, 
        input = chunks
    )
    return response

# Create IndexIVFFlat
def create_indexIVF(file_path, folder_path, graph = False):
    '''
    Creates a Faiss index for a specific set of documents.
    Args:   file_path (str): path to the folder of documents to be indexed.
            folder_path (str): intended folder for the .faiss and .pkl files
            graph (bool): a graph showing the number of documents indexed versus elapsed time
    Returns: the created file store
    '''

    # Prepare data
    local_file_path = file_path
    path = Path(folder_path)
    path.mkdir(exist_ok=True, parents=True)

    # Create embedding client
    client = AzureOpenAI(
        azure_deployment = MODEL_DEPLOYMENT_NAME, 
        api_version = MODEL_API_VERSION,
        azure_endpoint = os.environ["OPENAI_MODEL_ENDPOINT"],
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
            new_embeddings = []
            with open(each_file_path, "r", encoding="utf-8") as f:
                title = f.readline().strip('\n')
            
            # Create embeddings
            chunks_list = get_chunks(each_file_path)
            new_chunks = [chunks_list[i * MAX_ARRAY_SIZE:(i + 1) * MAX_ARRAY_SIZE] for i in range((len(chunks_list) + MAX_ARRAY_SIZE - 1) // MAX_ARRAY_SIZE)]  
            
            # Split chunks array into size 2048 max
            for i in range(len(new_chunks)):
                response = create_embeddings(client, new_chunks[i])
                for j in range(len(new_chunks[i])):
                    new_embeddings.append(np.array(response.data[j].embedding))
            embeddings.extend(new_embeddings)
            chunks.extend(chunks_list)
            count = len(new_embeddings)
            metadatas.extend([
                {"title": title, "source": os.path.join(file)}
            ] * count)

            # Update progress bar and graph
            progress_bar.update()
            if graph:
                elapsed_times.append(progress_bar.format_dict['elapsed'])

    # Create array of embeddings
    xb = np.array(embeddings)

    # Update Docstore
    documents = [Document(page_content=t, metadata=m) for t, m in zip(chunks, metadatas)]

    # Train index and add embeddings to it
    quantizer = faiss.IndexFlatL2(DIMENSION)
    index_file = Path(str(path / "index.faiss"))
    pkl_file = Path(str(path / "index.pkl"))

    if not index_file.is_file() or not pkl_file.is_file():
        print("Index does not exist")
        index = faiss.IndexIVFFlat(quantizer, DIMENSION, NLIST)
        index.train(xb)
        index.add(xb)
        index_to_docstore_id = {i: str(uuid.uuid4()) for i in range(len(documents))}
        docstore = InMemoryDocstore({index_to_docstore_id[i]: doc for i, doc in enumerate(documents)})
    else:
        print("Index exists")
        index = faiss.read_index(str(path / "index.faiss"))
        with open(str(path / "index.pkl"), 'rb') as f:
            docstore, index_to_docstore_id = pickle.load(f)
        if index.ntotal != len(index_to_docstore_id):
            return "Error: Current .index and .pkl files do not have the same length."
        else:
            index.add(xb)
            starting_len = len(index_to_docstore_id)
            index_to_id = {starting_len + j: str(uuid.uuid4()) for j in range(len(documents))}
            index_to_docstore_id.update(index_to_id)
            docstore.add({index_to_id[starting_len + j]: doc for j, doc in enumerate(documents)})

    # Write index to local file
    print("Index size:", index.ntotal)
    print("Pkl size:", len(index_to_docstore_id))
    faiss.write_index(index, str(path / "index.faiss"))
    with open(path / "index.pkl", "wb") as f:
        pickle.dump((docstore, index_to_docstore_id), f)

    # Close progress bar and create line graph
    progress_bar.close()
    if graph: 
        x = np.array(range(progress_bar.format_dict['total']))
        y = np.array(elapsed_times)
        plt.plot(x, y) 
        plt.xlabel("# of Documents Indexed") 
        plt.ylabel("Elapsed Time (seconds)")
        plt.title("# of Documents Indexed versus Elapsed Time")
        plt.show()
    return index