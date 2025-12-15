from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.logger import logger

class Chunker:
    def __init__(self,chunk_size:int = 400,chunk_overlap:int = 100):
        self.splitter = RecursiveCharacterTextSplitter(separators=[
                                                                    "\n\n",
                                                                    "\n",
                                                                    ". ",
                                                                    "! ",
                                                                    "? ",
                                                                    ", ",
                                                                    " ",
                                                                    ""
                                                                    ],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       )
        
    def split_text(self,text):
        if text == "":
            chunks=[]
            logger.warning("⚠️ Le texte fourni est vide. Le chunks retourné est vide (chunks = [])")
            return chunks
        
        chunks = self.splitter.split_text(text)
        return chunks
    