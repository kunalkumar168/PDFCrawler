import os
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM

class ChatBot:
    def __init__(self, model_name: str, embedding_model_name: str):
        self.curr_workdir = os.getcwd()
        self.embedding_model_name = embedding_model_name
        self.model_name = model_name

        print("[INFO] Initializing ChatBot...")
        self._load_vector_db()
        self._load_llm()
        self._create_prompt_templates()

    def _load_vector_db(self):
        """Load the FAISS vector database with the specified embedding model."""
        print(f"[INFO] Loading vector DB from {self.curr_workdir}/vectordb ...")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.db = FAISS.load_local(
            f"{self.curr_workdir}/vectordb",
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("[INFO] Vector DB loaded successfully.")

    def _load_llm(self):
        """Load the Ollama model."""
        print(f"[INFO] Loading Ollama model: {self.model_name} ...")
        self.llm = OllamaLLM(model=self.model_name)
        print("[INFO] LLM loaded successfully.")

    def _create_prompt_templates(self):
        """Define the QA and validation prompt templates."""
        self.qa_prompt_template = PromptTemplate(
                input_variables = ["context", "question"],
                template = """
                            You are an intelligent assistant helping to answer questions based on provided information. Use only the context provided to answer the question.

                            Context:
                            {context}

                            Question:
                            {question}

                            Provide short and concise answer as per the question. If the answer is not in the context, say "Can't find Information".
                            """
            )

        self.ans_prompt_template = PromptTemplate(
                input_variables = ["expected", "predicted"],
                template = """
                            You are an intelligent assistant helping to validate the predicted answer by comparing it to the expected answer.

                            Expected Answer:
                            {expected}

                            Predicted Answer:
                            {predicted}

                            Please respond with "Yes" if the full or part of the Expected Answer is present in the Predicted Answer. Otherwise, respond with "No". Don't print anything else than "Yes" or "No".
                            """
            )

    def response(self, query: str, top_k: int = 3):
        """Get an answer to a query using the RAG pipeline."""
        print(f"[QUERY] {query}")
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_prompt_template}
        )

        result_obj = qa_chain({"query": query})
        answer = result_obj["result"]
        sources = [
            doc.metadata.get("source", "unknown").split("/")[-1]
            for doc in result_obj["source_documents"]
        ]

        return answer.strip(), sources

    def validate_answers(self, expected: str, predicted: str):
        """Check if the predicted answer is valid given the expected answer."""
        prompt = self.ans_prompt_template.format(
            expected=expected.lower(),
            predicted=predicted.lower()
        )
        response = self.llm(prompt)
        return response.strip()
