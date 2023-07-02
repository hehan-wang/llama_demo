from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, \
    ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
import sys
import os

os.environ['OPENAI_API_KEY'] = "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"
# os.environ["http_proxy"] = "https://proxy.hehanwang.com"
# os.environ["https_proxy"] = "https://proxy.hehanwang.com"


def create_index(path):
    max_input = 4096

    tokens = 2048
    chunk_size = 600  # forLLM,weneedtodefinechunksize
    max_chunk_overlap = 20
    # define prompt
    promptHelper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)
    # define LLM - there could be many models we can use , but in this example, let's go with OpenAI model
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))

    # load data - it will take all the . txtx files , if there are more than 1
    docs = SimpleDirectoryReader(path).load_data()

    # create vector index
    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=promptHelper)
    vectorIndex = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context)
    vectorIndex.storage_context.persist(persist_dir='store')


def answer(question_txt):
    storage_context = StorageContext.from_defaults(persist_dir='Store')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(question_txt)
    return response


# main
if __name__ == "__main__":
    print("Welcome to LLAMA!")
    # 读取当前目录下的doc目录
    # create_index("./doc")
    print(answer("小米为什么造车"))
