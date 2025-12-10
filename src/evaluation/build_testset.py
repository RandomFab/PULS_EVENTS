from src.rag.retriever import SimpleRetriever
from config.config import (
    INDEX_PATH,
    MISTRAL_API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL,
    TEST_QUESTION_PATH,
    TESTSET_PATH
)
import json
from config.logger import logger


def build_testset():
    rag = SimpleRetriever(
        INDEX_PATH,
        api_key=MISTRAL_API_KEY,
        embedding_model=EMBEDDING_MODEL,
        model_name=MODEL_NAME
    )
    rag.indexer.load()

    with open(TEST_QUESTION_PATH, "r", encoding='utf-8') as f:
        test_questions = json.load(f)

    dataset = []
    for q in test_questions:
        question = q['question']
        ground_truth = q['ground_truth']

        docs = rag.indexer.search(question, k=5)
        retrieved_contexts = [doc[0].page_content for doc in docs]
        answer = rag.answer_query(question)

        item = {
            'question': question,
            'ground_truth': ground_truth,
            'contexts': retrieved_contexts,  # For Faithfulness and AnswerRelevancy
            'retrieved_contexts': retrieved_contexts,  # For ContextPrecision and ContextRecall
            'answer': answer
        }

        dataset.append(item)

    with open(TESTSET_PATH, "w", encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    logger.info("✅ Dataset d'évaluation généré avec succès")


if __name__ == "__main__":
    build_testset()