import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def main():
    llm = ChatOpenAI(model_name="gpt-4o")
    confluence_api_key = os.environ.get("CONFLUENCE_API_TOKEN")
    confluence_user_name = os.environ.get("CONFLUENCE_USER_NAME")
    print(confluence_api_key)
    loader = ConfluenceLoader(
        url="https://athumi.atlassian.net/wiki",
        username=confluence_user_name,
        api_key=confluence_api_key,
        # cloud=False
    )
    docs = loader.load(space_key="BURE", include_attachments=True, limit=50)

    print(docs)

    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    # document_chain.invoke({
    #     "input": "how can langsmith help with testing?",
    #     "context": [Document(page_content="langsmith can let you visualize test results")]
    # })

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({
        "input": "Waar staat VIP voor en wat is het doel van dit platform?"})
    print(response["answer"])


if __name__ == "__main__":
    main()
