from typing import List

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import SitemapLoader,WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def main():
    llm = ChatOpenAI(model_name="gpt-4o")
    docs = []
    countries = get_countries()
    for country in countries:
        loader = WebBaseLoader(country)
        loader.session.headers['User-Agent'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
        docs = docs + loader.load()

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

    response = retrieval_chain.invoke({"input": "How much would it cost to travel for 4 weeks to albania with a budget travel style in Euro when we don't have to pay for sleep accomodation since we will travel by camper?"})
    print(response["answer"])

def get_countries()->list[str]:
    return [
        "https://www.budgetyourtrip.com/albania",
        "https://www.budgetyourtrip.com/croatia",
        "https://www.budgetyourtrip.com/georgia",
        "https://www.budgetyourtrip.com/bulgaria",
        "https://www.budgetyourtrip.com/budgetreportadv.php?geonameid=&countrysearch=&country_code=AL&categoryid=0&budgettype=1&triptype=0&startdate=&enddate=&travelerno=0"
    ]


if __name__ == "__main__":
    main()

