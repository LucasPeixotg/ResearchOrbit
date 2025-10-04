from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from orbitsAPI.src.rag.settings import settings
import asyncio
import ollama
import fitz  # PyMuPDF
import re
import os

class RAG:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _clean_pdf_text(self, text: str) -> str:
        """
        Cleans PDF text:
        - Remove headers/footers (simple heuristic: lines repeated many times)
        - Dehyphenate words split across lines
        - Normalize whitespace
        """
        # Normalize newlines and whitespace
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{2,}', '\n', text)  # collapse multiple blank lines

        # Remove hyphenation at line breaks: "exam-\nple" -> "example"
        text = re.sub(r'-\n', '', text)

        # Remove remaining line breaks inside paragraphs
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Optional: remove repeated headers/footers (simple heuristic)
        lines = text.split('\n')
        lines_seen = set()
        cleaned_lines = []
        for line in lines:
            line_strip = line.strip()
            if line_strip and line_strip not in lines_seen:
                cleaned_lines.append(line_strip)
                lines_seen.add(line_strip)
        text = ' '.join(cleaned_lines)

        return text

    def _load_and_clean_pdfs(self) -> list[Document]:
        """
        Loads PDFs, cleans text, splits into chunks, and returns LangChain Documents with rich metadata
        """
        directory = settings.documents_directory
        all_documents = []

        for filename in os.listdir(directory):
            if not filename.lower().endswith(".pdf"):
                continue

            filepath = os.path.join(directory, filename)
            pdf = fitz.open(filepath)

            title = pdf.metadata.get('title', filename)

            total_pages = pdf.page_count
            full_text = ""

            # Extract all pages
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text()
                full_text += page_text + "\n"
            pdf.close()

            cleaned_text = self._clean_pdf_text(full_text)

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,          # maximum tokens/characters per chunk
                chunk_overlap=settings.chunk_overlap,    # overlap between chunks
                separators=settings.splitter_separators  # try larger separators first
            )
            chunks = splitter.split_text(cleaned_text)

            # Convert chunks into Documents with metadata
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": title,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "total_pages": total_pages
                    }
                )
                all_documents.append(doc)

        return all_documents
    
    def __init__(self):
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
            model_kwargs={'device': settings.device}
        )
        self.docs = self._load_and_clean_pdfs()

        # Create FAISS retriever
        # Add caching for embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
            model_kwargs={'device': settings.device},
            encode_kwargs={'normalize_embeddings': True}  # Faster similarity search
        )
        
        # Cache the FAISS index to disk
        index_path = "faiss_index"
        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(
                index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            vectorstore = FAISS.from_documents(self.docs, self.embeddings)
            vectorstore.save_local(index_path)
            
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.faiss_k})
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(self.docs)
        bm25_retriever.k = settings.bm25_k
        
        # Create ensemble retriever combining both
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=settings.ensemble_weights  # Equal weights; adjust based on your needs
        )

        # Initialize cross-encoder for reranking
        self.reranker = CrossEncoder(
            settings.rerank_model,
            max_length=512,
            device=settings.device
        )

    def _generate_hyde_prompt(self, prompt: str) -> str:
        response = ollama.generate(
            model=settings.hyde_model,
            prompt=f"""Generate a short, concise paragraph that reads like part of a factual article or encyclopedia entry. It should be 2-3 sentences maximum.

Question: {prompt}

Answer:""",
            stream=False,
            options={
                'temperature': settings.hyde_temp,
                'top_k': settings.hyde_topk,
                'top_p': settings.hyde_topp,
                'num_predict': settings.num_pred
            }
        )

        return response['response']

    def _retrieve_documents(self, prompt: str):
        """
        Retrieve relevant documents from the vectorstore for a given prompt.
        Returns a list of LangChain Document objects.
        """
        if not hasattr(self, "retriever"):
            raise ValueError("Retriever is not initialized. Make sure __init__ has been called.")

        # Use the retriever to get top-k documents
        docs = self.retriever.invoke(prompt)
        return docs

    def _rerank_documents(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Rerank documents using a cross-encoder model for better relevance.
        
        Args:
            query: The search query/question
            documents: List of retrieved documents to rerank
            top_k: Number of top documents to return after reranking (default: settings.rerank_top_k)
        
        Returns:
            List of reranked documents (top_k most relevant)
        """
        if not documents:
            return []
        
        # Prepare query-document pairs for the cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get relevance scores from cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Sort documents by score (descending)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents with scores added to metadata
        reranked_docs = []
        for doc, score in doc_score_pairs[:settings.top_k]:
            # Create a new document with added rerank score
            new_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    'rerank_score': float(score)
                }
            )
            reranked_docs.append(new_doc)
        
        return reranked_docs

    def _generate_context_from_documents(self, documents: list[Document]) -> str:
        context = ""
        
        for i, doc in enumerate(documents):
            source = doc.metadata['source']
            content = doc.page_content
            context += f"[id: doc{i} | source: {source}]\n{content}\n\n"
        
        return context 

    def _generate_context_based_answer(self, documents: list[Document], prompt: str, stream: bool=False) -> str:
        context = self._generate_context_from_documents(documents)

        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the user's question based on the provided context"
                },
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document IDs used to generate the answer (e.g., ['doc0', 'doc1'])"
                }
            },
            "required": ["answer", "source_ids"]
        }

        prompt = f"""Using ONLY the following context, answer the user's question.
After your answer, list the document IDs you used (format: Used sources: id1, id2, id3)

Context: 
{context}

Question: {prompt}

Answer:"""
        
        print(f"PROMPT: \n {prompt}")

        response = ollama.generate(
            model=settings.answer_model,
            prompt=f"""Generate a short, concise paragraph that reads like part of a factual article or encyclopedia entry. It should be 2-3 sentences maximum.

Question: {prompt}

Answer:""",
            stream=stream,
            format=schema,
            options={
                'temperature': settings.answer_temp,
                'top_k': settings.answer_topk,
            }
        )

        return response['response']
    

    def prompt(self, prompt: str) -> str:
        hyde_prompt = self._generate_hyde_prompt(prompt)
        # print(f"HYDE PROMPT: {hyde_prompt}")

        relevant_docs = self._retrieve_documents(hyde_prompt)

        # Rerank with the original user prompt (not hyde_prompt)
        reranked_docs = self._rerank_documents(prompt, relevant_docs)
        
        # print(f"\n{'='*50}")
        # print(f"RERANKED DOCUMENTS (Top {len(reranked_docs)})")
        # print(f"{'='*50}\n")
        
        # for d in reranked_docs:
        #     print(14*"=")
        #     print(f"SOURCE: {d.metadata['source']}")
        #     print(d.page_content)

        response = self._generate_context_based_answer(reranked_docs, prompt)

        return response
    
    # In rag.py, change back to:
    # def prompt_stream(self, prompt: str):  # Remove async
    #     hyde_prompt = self._generate_hyde_prompt(prompt)
    #     relevant_docs = self._retrieve_documents(hyde_prompt)
    #     reranked_docs = self._rerank_documents(prompt, relevant_docs)
        
    #     response_generator = self._generate_context_based_answer(reranked_docs, prompt, stream=True)
        
    #     for chunk in response_generator:
    #         yield chunk