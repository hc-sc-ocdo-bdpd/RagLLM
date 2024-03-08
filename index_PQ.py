from faiss_index import FaissIndex
import faiss

class IndexPQ(FaissIndex):
    def __init__(self) -> None:
        super().__init__()
        self.M = 8
        self.NBITS = 8

    def create(self, input_path: str, output_path: str, graph: bool = False):
        index_constructor = faiss.IndexPQ(self.DIMENSION, self.M, self.NBITS)
        super().create(index_constructor, input_path, output_path, graph)