import os
import random
import string

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from tavily import TavilyClient

from langchain_llm_objects import retrieval_grader_prompt, llm_llama3_json_temp0, llm_llama3_temp0, \
    generate_answer_prompt, router_prompt, hallucination_grader_prompt
from langchain_graph import GraphState, make_web_search_node, make_retriever_node, push_vectorstore, \
    make_generator_node, make_route_question_edge, decide_to_generate, \
    make_grade_generation_v_documents_and_question_edge
from typing import List

from langgraph.graph import StateGraph

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


def generate_random_string(length: int) -> str:
    characters = string.ascii_letters
    random_string = ''.join(random.choices(characters, k=length))
    return random_string


def execute(
        question: str,
        doc_urls: List[str],
        openai_api_key: str = OPENAI_API_KEY,
        langchain_api_key: str = LANGCHAIN_API_KEY,
        tavily_api_key: str = TAVILY_API_KEY,
) -> str:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

    tavily = TavilyClient(tavily_api_key)
    retrieval_grader = retrieval_grader_prompt | llm_llama3_json_temp0 | JsonOutputParser()
    generate_answer_chain = generate_answer_prompt | llm_llama3_temp0 | StrOutputParser()
    question_router = router_prompt | llm_llama3_json_temp0 | JsonOutputParser()
    hallucination_grader = hallucination_grader_prompt | llm_llama3_json_temp0 | JsonOutputParser()

    retriever = push_vectorstore(doc_urls, generate_random_string(20))

    workflow = StateGraph(GraphState)

    workflow.add_node("websearch", make_web_search_node(tavily))
    workflow.add_node("retrieve", make_retriever_node(retriever))
    workflow.add_node("grade_documents", make_generator_node(retrieval_grader))
    workflow.add_node("generate", make_generator_node(generate_answer_chain))

    workflow.set_conditional_entry_point(
        make_route_question_edge(question_router),
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        make_grade_generation_v_documents_and_question_edge(hallucination_grader),
        {
            "not supported": "generate",
            "useful": "__end__",
            "not useful": "websearch",
        },
    )

    app = workflow.compile()

    inputs = {"question": question}
    result = ""
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")
            if "generation" in value:
                result = value["generation"]

    return result
