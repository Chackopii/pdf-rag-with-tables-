# import required dependencies
# https://docs.chainlit.io/integrations/langchain
import os
from langchain_community.llms.ollama import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_qdrant import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient

import logging
logging.getLogger().setLevel(logging.ERROR)

import chainlit as cl
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {input}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
chat_history = []

client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url,)

llm = Ollama(model="zephyr:7b-alpha-q4_1",)

def create_documents_chain():
    """
    Prompt template for QA retrieval for each vectorstore
    """

    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'input'])
    
    documents_retriever= create_stuff_documents_chain(llm, prompt)

    return documents_retriever

def create_history_chain(retriever):
    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Given the above conversation, generation a search query to lookup in order to get information relevant to the conversation",
            ),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, retriever_prompt)

    return history_aware_retriever
@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    
    embeddings = FastEmbedEmbeddings()

    vectorstore = Qdrant(client=client, embeddings=embeddings, collection_name="rag")

    retriever=vectorstore.as_retriever(search_kwargs={'k': 2})

    history_chain=create_history_chain(retriever)

    documents_chain=create_documents_chain()

    rag_chain = create_retrieval_chain(history_chain, documents_chain)

    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Llamaparse, LangChain, Qdrant and open source model zephyr 7b alpha."
    )
    await welcome_message.update()


    cl.user_session.set("chain", rag_chain)


@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    from langchain.schema.runnable.config import RunnableConfig
    chain = cl.user_session.get("chain")
    print(message.content)
    msg = cl.Message(content="")
    print(msg)
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True,)
    config = RunnableConfig(callbacks=[cb])
    result = await chain.ainvoke({"input": message.content}, config=config) 
    print(result["answer"])
    

   
    chat_history.append(HumanMessage(content=message.content))
    chat_history.append(AIMessage(content=result["answer"]))

    answer = result["answer"]
    source_documents = result["context"]

    text_elements = []  ##type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()