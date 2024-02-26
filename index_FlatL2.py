from faiss_index import FaissIndex
import faiss

class IndexFlatL2(FaissIndex):
    def __init__(self, file_path: str, folder_path: str, graph: bool = False) -> None:
        super().__init__(file_path, folder_path, graph)

    def create(self):
        """Creates a FlatL2 Faiss index for a specific set of documents."""
        index_constructor = faiss.IndexFlatL2(self.DIMENSION)
        super().create(index_constructor)