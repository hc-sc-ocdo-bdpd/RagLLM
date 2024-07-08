from faiss_index import FaissIndex
import faiss

class IndexHNSW(FaissIndex):
    def __init__(self) -> None:
        super().__init__()

    def create(self, input_path: str, output_path: str, M: int):
        self.M = M
        index_constructor = faiss.IndexHNSWFlat(self.DIMENSION, self.M)
        super().create(index_constructor, input_path, output_path)