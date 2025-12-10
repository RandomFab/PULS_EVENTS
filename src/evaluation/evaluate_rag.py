from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
    Faithfulness
)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from datasets import Dataset
import json
import os
from openai import OpenAI
from mistralai import Mistral

from config.config import TESTSET_PATH, MISTRAL_API_KEY


def ragas_evaluation():
    # Disable Ragas telemetry to avoid OpenAI dependency
    os.environ['RAGAS_DO_NOT_TRACK'] = '1'
    
    # Set dummy OpenAI key to prevent errors (won't be used)
    os.environ['OPENAI_API_KEY'] = 'dummy-key'

    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    ds = Dataset.from_list(data)

    # Use OpenAI-compatible client for Mistral LLM
    client = OpenAI(
        api_key=MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1"
    )
    llm = llm_factory('mistral-large-latest', client=client)

    # Use embedding_factory with Mistral client
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    embeddings = embedding_factory('mistral-embed', client=mistral_client)

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
