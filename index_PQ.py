from faiss_index import FaissIndex
import faiss

class IndexPQ(FaissIndex):
    def __init__(self, file_path: str, folder_path: str, graph: bool = False) -> None:
        super().__init__(file_path, folder_path, graph)
        self.M = 8
        self.NBITS = 8

    def create(self):
        """Creates a PQ Faiss index for a specific set of documents."""
        index_constructor = faiss.IndexPQ(self.DIMENSION, self.M, self.NBITS)
        super().create(index_constructor)