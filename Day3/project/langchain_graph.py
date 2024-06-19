from pprint import pprint
from typing import List, Callable

from langchain_core.documents import Document
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores import VectorStoreRetriever
from tavily import TavilyClient
from typing_extensions import TypedDict

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]


def push_vectorstore(doc_urls: List[str], collection_name: str) -> VectorStoreRetriever:
    docs = [WebBaseLoader(url).load() for url in doc_urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        persist_directory="./db_chroma/chroma_db",
        documents=doc_splits,
        collection_name=collection_name,
        embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
    )

    return vectorstore.as_retriever()


def make_retriever_node(retriever: VectorStoreRetriever, answer_container) -> Callable[[any], dict]:
    def retrieve(state):
        # answer_container.write("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    return retrieve


def make_generator_node(rag_chain: RunnableSerializable, answer_container) -> Callable[[any], dict]:
    def generate(state):
        # answer_container.write("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        generation = rag_chain.invoke({"documents": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    return generate


def make_grade_documents_node(retrieval_grader: RunnableSerializable, answer_container) -> Callable[[any], dict]:
    def grade_documents(state):
        # answer_container.write("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            if grade.lower() == "yes":
                # answer_container.write("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)

            else:
                # answer_container.write("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    return grade_documents


def make_web_search_node(tavily: TavilyClient, answer_container) -> Callable[[any], dict]:
    def web_search(state):
        # answer_container.write("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = tavily.search(query=question)['results']

        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}

    return web_search


### Edges


def make_route_question_edge(question_router: RunnableSerializable, answer_container) -> Callable[[any], str]:
    def route_question(state):
        # answer_container.write("---ROUTE QUESTION---")
        question = state["question"]
        source = question_router.invoke({"question": question})

        if source["datasource"] == "web_search":
            return "websearch"
        elif source["datasource"] == "vectorstore":
            return "vectorstore"

    return route_question


def make_decide_to_generate(answer_container) -> Callable[[any], str]:
    def decide_to_generate(state):
        # answer_container.write("---ASSESS GRADED DOCUMENTS---")
        web_search = state["generation"]

        if web_search == "Yes":
            # answer_container.write("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            # answer_container.write("---DECISION: GENERATE---")
            return "generate"

    return decide_to_generate


def make_grade_generation_v_documents_and_question_edge(
        hallucination_grader: RunnableSerializable,
        # answer_grader: RunnableSerializable,
        answer_container
):
    def grade_generation_v_documents_and_question(state):
        # answer_container.write("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            # answer_container.write("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
            # score = answer_grader.invoke({"question": question, "generation": generation})
            # grade = score["score"]
            # if grade == "yes":
            #     return "useful"
            # else:
            #     return "not useful"
        else:
            # answer_container.write("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    return grade_generation_v_documents_and_question
