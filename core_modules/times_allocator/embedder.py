from core_modules.times_allocator import embedding_trainer as et
from core_modules.times_allocator import embedding_trainer_act_weighting_no_times as etawnt
from core_modules.times_allocator import embedding_trainer_act_weighting_times as etawt
from core_modules.times_allocator import embedding_trainer_times as ett
from core_modules.times_allocator import embedding_word2vec as ew
from support_modules.common import EmbeddingMethods as Em


class Embedder:
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, params, log, ac_index, index_ac, usr_index, index_usr):
        """constructor"""
        self.log = log.copy()
        self.params = params
        print(params)
        self.ac_index = ac_index
        self.index_ac = index_ac
        self.usr_index = usr_index
        self.index_usr = index_usr
        self.file_name = params['file']
        self.embedded_path = params['embedded_path']
        self.include_times = params['include_times']
        self.embedding_file_name = None

    def create_embeddings(self, method):
        embedder_class = self._get_embedder(method)
        embedder = embedder_class(self.params, self.log, self.ac_index, self.index_ac, self.usr_index, self.index_usr)
        self.embedding_file_name = embedder.embedding_file_name
        return embedder.load_embeddings()

    def _get_embedder(self, method):
        if method == Em.DOT_PROD:
            return et.EmbeddingTrainer
        elif method == Em.W2VEC:
            return ew.EmbeddingWord2vec
        elif method == Em.DOT_PROD_TIMES:
            return ett.EmbeddingTrainer
        elif method == Em.DOT_PROD_ACT_WEIGHT and self.include_times:
            return etawt.EmbeddingTrainer
        elif method == Em.DOT_PROD_ACT_WEIGHT and not self.include_times:
            return etawnt.EmbeddingTrainer
        else:
            raise ValueError(method)
