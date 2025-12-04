# Fichier de configuration\n# Contient les cl�s API et param�tres globaux.
from dotenv import load_dotenv
import os
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY")
