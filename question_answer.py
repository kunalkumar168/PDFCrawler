import json
import os
import pandas as pd
from chatbot import ChatBot
import string
import re
from sklearn.metrics import f1_score
from math import log2
from create_relevance import Relevance

class EvalMetrics:
    def __init__(self):
        pass

    def _normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)  # remove articles
        text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
        return text

    def compute_exact_match(self, predicted, expected):
        return self._normalize_text(predicted) == self._normalize_text(expected)

    def compute_f1(self, predicted, expected):
        pred_tokens = self._normalize_text(predicted).split()
        exp_tokens = self._normalize_text(expected).split()
        common = set(pred_tokens) & set(exp_tokens)
        if len(common) == 0:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(exp_tokens)
        return 2 * precision * recall / (precision + recall)

class RelevanceScore(Relevance):
    def __init__(self, model_name, k):
        super().__init__(model_name)
        self.k = k

    def precision_at_k(self, relevant_flags):
        return sum(relevant_flags[:self.k]) / self.k

    def dcg_at_k(self, relevant_flags):
        return sum([(2**rel - 1) / log2(idx + 2) for idx, rel in enumerate(relevant_flags[:self.k])])

    def ndcg_at_k(self, relevant_flags):
        ideal = sorted(relevant_flags[:self.k], reverse=True)
        dcg = self.dcg_at_k(relevant_flags)
        idcg = self.dcg_at_k(ideal)
        return dcg / idcg if idcg > 0 else 0.0


def get_answers(query_data, model_name, embedding_model_name):
    results = []
    top_k = 10
    chatbot = ChatBot(model_name, embedding_model_name)
    relevance = RelevanceScore(embedding_model_name, top_k)
    metric = EvalMetrics()

    for item in query_data['queries']:
        query = item['question'].lower()
        expected = item['answer']
        
        predicted, sources = chatbot.response(query, top_k)
        source_names = [ source['source'] for source in sources ]
        source_content = [ source['page_content'] for source in sources ]
        answer_matches = chatbot.validate_answers(expected, predicted)
        
        relevant_flags = [ relevance.get_relevancy(chunk, expected) for chunk in source_content ]
        precision_at_k = relevance.precision_at_k(relevant_flags)
        ndcg_at_k = relevance.ndcg_at_k(relevant_flags)

        em = metric.compute_exact_match(predicted, expected)
        f1 = metric.compute_f1(predicted, expected)
        match_flag = 1 if 'yes' in answer_matches.lower() else 0

        results.append([query, expected, predicted, answer_matches, source_names, precision_at_k, ndcg_at_k, em, f1, match_flag])

    results_df = pd.DataFrame(results, columns=[
        'query', 'expected', 'predicted', 'answer_matches', 'sources', 'Precision@K', 'nDCG@K', 'exact_match', 'f1_score', 'match_flag'
    ])

    return results_df

def main():
    current_workdir = os.getcwd()
    query_path = f"{current_workdir}/query.json"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_name = "mistral"

    with open(query_path, "r") as file:
        query_data = json.load(file)

    results_df = get_answers(query_data, model_name, embedding_model_name)
    results_df.to_csv(f'{current_workdir}/results_new.csv', index=False)
    print(f"\n✅ Results saved to: {current_workdir}/results_new.csv")

if __name__ == '__main__':
    main()