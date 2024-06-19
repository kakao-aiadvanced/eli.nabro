import os
import random
import string

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from tavily import TavilyClient

from langchain_llm_objects import retrieval_grader_prompt, llm_llama3_json_temp0, llm_llama3_temp0, \
    generate_answer_prompt, router_prompt, hallucination_grader_prompt, search_engine_prompt
from langchain_graph import GraphState, make_web_search_node, make_retriever_node, push_vectorstore, \
    make_generator_node, make_route_question_edge, make_decide_to_generate, \
    make_grade_generation_v_documents_and_question_edge, make_search_engine_node, decide_engine_edge
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
        answer_container,
        question: str,
        doc_urls: List[str],
        openai_api_key: str = OPENAI_API_KEY,
        langchain_api_key: str = LANGCHAIN_API_KEY,
        tavily_api_key: str = TAVILY_API_KEY,
):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

    tavily = TavilyClient(tavily_api_key)
    retrieval_grader = retrieval_grader_prompt | llm_llama3_json_temp0 | JsonOutputParser()
    generate_answer_chain = generate_answer_prompt | llm_llama3_temp0 | StrOutputParser()
    question_router = router_prompt | llm_llama3_json_temp0 | JsonOutputParser()
    hallucination_grader = hallucination_grader_prompt | llm_llama3_json_temp0 | JsonOutputParser()
    search_engine_chooser = search_engine_prompt | llm_llama3_temp0 | StrOutputParser()

    retriever = push_vectorstore(doc_urls, generate_random_string(20))

    workflow = StateGraph(GraphState)

    workflow.add_node("websearch", make_web_search_node(tavily, answer_container))
    workflow.add_node("retrieve", make_retriever_node(retriever, answer_container))
    workflow.add_node("grade_documents", make_generator_node(retrieval_grader, answer_container))
    workflow.add_node("generate", make_generator_node(generate_answer_chain, answer_container))
    workflow.add_node("search_engine_chooser", make_search_engine_node(search_engine_chooser))

    workflow.set_conditional_entry_point(
        make_route_question_edge(question_router, answer_container),
        {
            "websearch": "search_engine_chooser",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("websearch", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        make_decide_to_generate(answer_container),
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )

    workflow.add_conditional_edges(
        "search_engine_chooser",
        decide_engine_edge,
        {
            "tavily": "websearch",
            "arxiv": "websearch"
        }
    )

    workflow.add_conditional_edges(
        "generate",
        make_grade_generation_v_documents_and_question_edge(hallucination_grader, answer_container),
        {
            "not supported": "generate",
            "useful": "__end__",
            "not useful": "websearch",
        },
    )

    app = workflow.compile()

    inputs = {"question": question}
    result = ""
    result_urls = []
    for output in app.stream(inputs):
        for key, value in output.items():
            answer_container.write(f"Finished running: {key}:")
            context = answer_container.popover(f"  {key} context")
            context.write(f"  {value}")
            if "generation" in value:
                result = value["generation"]

            if "documents" in value:
                documents = value["documents"]
                urls = [f"{d.metadata['title']} : {d.metadata['source']}" for d in documents]
                if len(urls) > 0:
                    result_urls = urls

    return result, str(list(set(result_urls)))
