from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import AzureOpenAI
import os
import numpy as np
import pickle
from tqdm import tqdm
from langchain.docstore.document import Document
import faiss
import uuid
from langchain.docstore.in_memory import InMemoryDocstore
import matplotlib.pyplot as plt 

class FaissIndex(ABC):
    def __init__(self, file_path: str, folder_path: str, graph: bool = False) -> None:
        self.file_path = file_path
        self.folder_path = Path(folder_path)
        self.has_graph = graph
        self.MODEL_API_VERSION = "2023-05-15"
        self.MODEL_DEPLOYMENT_NAME = "ada_embedding"
        self.DIMENSION = 1536
        self.MAX_ARRAY_SIZE = 2048
        self.client = AzureOpenAI(
            azure_deployment = self.MODEL_DEPLOYMENT_NAME, 
            api_version = self.MODEL_API_VERSION,
            azure_endpoint = os.environ["OPENAI_MODEL_ENDPOINT"],
            api_key = os.environ["OPENAI_API_KEY"]
        )
        self.index = None
        self.docstore = None
        self.index_to_docstore_id = None

    def _get_chunks(self, file_name: str) -> List[str]:
        """Split documents into chunks."""
        with open(file_name, "r", encoding = "utf-8") as f:
            page_content = f.read()
            chunks = []
            splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 10)
            for chunk in splitter.split_text(page_content):
                chunks.append(chunk)
        return chunks
    
    @retry(wait = wait_random_exponential(min = 1, max = 60), stop = stop_after_attempt(6))
    def _create_embeddings(self, chunks: List[str]):
        """Create embeddings for each document chunk."""
        response = self.client.embeddings.create(
            model = self.MODEL_DEPLOYMENT_NAME, 
            input = chunks
        )
        return response

    @abstractmethod
    def create(self, index_constructor):
        """Abstract method to be implemented by subclasses."""
        # Create path to index file folder
        path = self.folder_path
        path.mkdir(exist_ok=True, parents=True)

        # Count the total number of files for the progress bar
        total_files = sum([len(files) for _, _, files in os.walk(self.file_path)])
        progress_bar = tqdm(total = total_files, desc = "Processing Files")
        elapsed_times, chunks, metadatas, embeddings = [], [], [], []
    
        for root, _, files in os.walk(self.file_path):
            for file in files:
                each_file_path = os.path.join(root, file)
                new_embeddings = []
                with open(each_file_path, "r", encoding="utf-8") as f:
                    title = f.readline().strip('\n')
                
                # Create chunks for each document
                chunks_list = self._get_chunks(each_file_path)
                
                # Split chunks array into size 2048 max
                new_chunks = [chunks_list[i * self.MAX_ARRAY_SIZE:(i + 1) * self.MAX_ARRAY_SIZE] for i in range((len(chunks_list) + self.MAX_ARRAY_SIZE - 1) // self.MAX_ARRAY_SIZE)]  
                
                # Create embeddings for the chunks array
                for i in range(len(new_chunks)):
                    response = self._create_embeddings(new_chunks[i])
                    for j in range(len(new_chunks[i])):
                        new_embeddings.append(np.array(response.data[j].embedding))
                
                # Record embeddings, chunks, count, and metadata
                embeddings.extend(new_embeddings)
                chunks.extend(chunks_list)
                count = len(new_embeddings)
                metadatas.extend([
                    {"title": title, "source": os.path.join(file)}
                ] * count)

                # Update progress bar
                progress_bar.update()
                if self.has_graph:
                    elapsed_times.append(progress_bar.format_dict['elapsed'])

        # Create array of embeddings
        xb = np.array(embeddings)
        
        # Create a document for each chunk
        documents = [Document(page_content=t, metadata=m) for t, m in zip(chunks, metadatas)]

        # Create paths to .faiss and .pkl files
        index_file = Path(str(path / "index.faiss"))
        pkl_file = Path(str(path / "index.pkl"))

        # Create or update index
        if not index_file.is_file() or not pkl_file.is_file():
            print("Index does not exist")
            index = index_constructor
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

        # Print file sizes (for testing)
        print("Index size:", index.ntotal)
        print("Pkl size:", len(index_to_docstore_id))

        # Write index to local files
        faiss.write_index(index, str(path / "index.faiss"))
        with open(path / "index.pkl", "wb") as f:
            pickle.dump((docstore, index_to_docstore_id), f)
    
        # Create line graph
        progress_bar.close()
        if self.has_graph:  
            x = np.array(range(progress_bar.format_dict['total']))
            y = np.array(elapsed_times)
            plt.plot(x, y) 
            plt.xlabel("# of Documents Indexed") 
            plt.ylabel("Elapsed Time (seconds)")
            plt.title("# of Documents Indexed versus Elapsed Time")
            plt.show()

        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id

    def read(self):
        index = faiss.read_index(str(self.folder_path / "index.faiss"))
        with open(str(self.folder_path / "index.pkl"), 'rb') as f:
            docstore, index_to_docstore_id = pickle.load(f)
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id

    def query(self, q: str):
        """Search the index with a query."""
        # Create query embedding
        response = self.client.embeddings.create(
            model = self.MODEL_DEPLOYMENT_NAME, 
            input = q
        )
        q_embedding = np.array([response.data[0].embedding])

        if self.index is not None:
            results = self.index.search(q_embedding, 3)
            docs = results[1][0]
            scores = results[0][0]
            for i in range(len(docs)):
                d = self.docstore.search(self.index_to_docstore_id.get(docs[i]))
                print(f"Source: {d.metadata.get('source')}")
                print(f"{d.metadata.get('title')}")
                print(f"Content: {d.page_content}")
                print(f"Score: {scores[i]}")
                print("=" * 50)
        else:
            return "Error: No index exists. Please use the create() function to make one."