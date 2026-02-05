from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader('./final_year_project_documentation.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)


from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import os

FAISS_DIR = './faiss_store'
EMBED_MODEL = 'embeddinggemma:300m'
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

def load_vector_store():
    if not os.path.exists(FAISS_DIR):
        vector_store = FAISS.from_documents(chunks, embeddings)
        FAISS.save_local(
            vector_store,
            folder_path=FAISS_DIR,
        )
        print("Store")
        return vector_store

    vector_store = FAISS.load_local(
        folder_path=FAISS_DIR,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    print("Local")
    return vector_store


# # vector_store = FAISS.from_documents(chunks, embeddings)
# # faiss.write_index(vector_store.index, 'index_file.faiss')
# vector_store = FAISS.load_local(faiss.read_index('index_file.faiss'))


vector_store = load_vector_store()
retriever = vector_store.as_retriever()


from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model='gemma3:4b')

def ask_on_document(question:str):
    context = retriever.invoke(question)
    context = [doc.page_content for doc in context]
    context = "\n\n".join(context)

    prompt = PromptTemplate.from_template("""
Answer the following question based on the provided context:
CONTEXT: 
<context>
{context}
</context>

QUESTION: "{question}"
""".strip()).format(
        context=context,
        question=question
    )

    print(
        # llm.invoke(prompt).content
        prompt
    )

ask_on_document('What is the project?')

# if __name__ == '__main__':
#     while True:
#         print("="*75)
#         usr = input("Ask on document: ")
#         result = qa_chain.run(usr)
#         print(result, end='\n\n')
