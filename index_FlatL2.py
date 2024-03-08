from faiss_index import FaissIndex
import faiss

class IndexFlatL2(FaissIndex):
    def __init__(self) -> None:
        super().__init__()

    def create(self, input_path: str, output_path: str, graph: bool = False):
        index_constructor = faiss.IndexFlatL2(self.DIMENSION)
        super().create(index_constructor, input_path, output_path, graph)