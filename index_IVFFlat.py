from faiss_index import FaissIndex
import faiss

class IndexIVFFlat(FaissIndex):
    def __init__(self) -> None:
        super().__init__()

    def create(self, input_path: str, output_path: str, n_list: int):
        self.NLIST = n_list
        quantizer = faiss.IndexFlatL2(self.DIMENSION)
        index_constructor = faiss.IndexIVFFlat(quantizer, self.DIMENSION, self.NLIST)
        super().create(index_constructor, input_path, output_path)