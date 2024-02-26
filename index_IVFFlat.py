from faiss_index import FaissIndex
import faiss

class IndexIVFFlat(FaissIndex):
    def __init__(self, file_path: str, folder_path: str, graph: bool = False) -> None:
        super().__init__(file_path, folder_path, graph)
        self.NLIST = 64

    def create(self):
        """Creates an IVFFlat Faiss index for a specific set of documents."""
        quantizer = faiss.IndexFlatL2(self.DIMENSION)
        index_constructor = faiss.IndexIVFFlat(quantizer, self.DIMENSION, self.NLIST)
        super().create(index_constructor)