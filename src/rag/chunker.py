from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self,chunk_size:int = 500,chunk_overlap:int = 150):
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
            return []
        return self.splitter.split_text(text)
    
