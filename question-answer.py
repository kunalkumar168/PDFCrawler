import json
import os
import pandas as pd
from chatbot import ChatBot

def get_answers(query_data, model_name, embedding_model_name):
    results = []
    for item in query_data['queries']:
        query = item['question'].lower()
        expected = item['answer']
        chatbot = ChatBot(model_name, embedding_model_name)
        predicted, sources = chatbot.response(query, top_k=3)
        answer_matches = chatbot.validate_answers(expected, predicted)
        results.append([query, expected, predicted, answer_matches, sources])
        break
    results_df = pd.DataFrame(results, columns=['query', 'expected', 'predicted', 'answer_matches', 'sources'])
    return results_df

def main():
    current_workdir = os.getcwd()
    query_path = f"{current_workdir}/query.json"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_name = "mistral"
    with open(query_path, "r") as file:
        query_data = json.load(file)
    results_df = get_answers(query_data, model_name, embedding_model_name)
    results_df.to_csv(f'{current_workdir}/results.csv', index=False)
    print(f"Your results are saved at : {current_workdir}/results.csv")

if __name__ == '__main__':
    main()