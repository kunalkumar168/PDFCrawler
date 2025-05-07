from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util


class Relevance:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = CountVectorizer().build_tokenizer()
        self._get_model()
    
    def _get_model(self):
        self.model = SentenceTransformer(self.model_name) 
    
    def is_relevant_chunk(self, chunk, expected_answer, threshold=0.5):
        chunk_embedding = self.model.encode(chunk, convert_to_tensor=True)
        answer_embedding = self.model.encode(expected_answer, convert_to_tensor=True)
        score = util.cos_sim(chunk_embedding, answer_embedding).item()
        return score >= threshold

    def keyword_overlap(self, chunk, expected_answer, threshold=0.5):
        chunk_tokens = set(self.tokenizer(chunk.lower()))
        answer_tokens = set(self.tokenizer(expected_answer.lower()))
        if not answer_tokens:
            return False
        overlap = chunk_tokens.intersection(answer_tokens)
        return (len(overlap) / len(answer_tokens)) >= threshold

    def get_relevancy(self, chunk, expected_answer):
        score1 = self.is_relevant_chunk(chunk, expected_answer)
        score2 = self.keyword_overlap(chunk, expected_answer)
        return 1 if (score1 or score2) else 0