from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
    Faithfulness
)
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
import json
import os
from openai import OpenAI
from langchain_mistralai import MistralAIEmbeddings

from config.config import TESTSET_PATH, MISTRAL_API_KEY


def ragas_evaluation():
    # Disable Ragas telemetry to avoid OpenAI dependency
    os.environ['RAGAS_DO_NOT_TRACK'] = '1'
    
    # Set dummy OpenAI key to prevent errors (won't be used)
    os.environ['OPENAI_API_KEY'] = 'dummy-key'

    # Check if testset file exists
    try:
        with open(TESTSET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Erreur lors du parsing du fichier JSON : {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier testset n'existe pas : {TESTSET_PATH}")
    
    if not data:
        raise ValueError("Le fichier testset est vide")
    
    ds = Dataset.from_list(data)

    # Use OpenAI-compatible client for Mistral LLM
    client = OpenAI(
        api_key=MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1"
    )
    llm = llm_factory('mistral-large-latest', client=client)

    # Wrap Mistral embeddings for Ragas
    mistral_embeddings = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)
    embeddings = LangchainEmbeddingsWrapper(mistral_embeddings)

    results = evaluate(
        dataset=ds,
        metrics=[
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            Faithfulness(llm=llm)
        ]
    )

    print(results)
    return results


if __name__ == "__main__":
    ragas_evaluation()
