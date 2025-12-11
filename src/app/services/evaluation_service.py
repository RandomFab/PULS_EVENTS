from src.evaluation.evaluate_rag import ragas_evaluation

def evaluate_current_rag():
    """
    Évalue le système RAG actuel en utilisant Ragas.
    
    Returns:
        dict: Résultats de l'évaluation ou message d'erreur
        
    Raises:
        FileNotFoundError: Si le fichier testset n'existe pas
        ValueError: Si le fichier JSON est invalide ou vide
        Exception: Pour toute autre erreur durant l'évaluation
    """
    try:
        results = ragas_evaluation()
        return {
            "status": "success",
            "message": "Évaluation terminée avec succès",
            "results": results
        }
    except FileNotFoundError as e:
        return {
            "status": "error",
            "error_type": "FileNotFoundError",
            "message": str(e),
            "details": "Le fichier de test n'existe pas. Vérifiez que build_testset.py a été exécuté."
        }
    except ValueError as e:
        return {
            "status": "error", 
            "error_type": "ValueError",
            "message": str(e),
            "details": "Erreur dans le format des données de test."
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": "Exception",
            "message": f"Erreur inattendue lors de l'évaluation : {str(e)}",
            "details": "Vérifiez la configuration Mistral AI et les dépendances."
        }
