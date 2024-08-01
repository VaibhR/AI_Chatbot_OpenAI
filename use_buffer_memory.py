# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'
from langchain_community.document_loaders import WebBaseLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI
output=[""]
temp_mem=[""]
loader = WebBaseLoader('https://www2.deloitte.com/us/en/insights/economy/asia-pacific/india-economic-outlook.html')
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)
# print(docs)
embeddings = OpenAIEmbeddings(api_key="----------------api-key----------")
db = FAISS.from_documents(docs, embeddings)
# print(db.index.ntotal)
# ques="what will be the importance of india in research or in future tech"
while True:
    ques = str(input("ask: "))
    answers = db.similarity_search(ques)
    # print(type(answers))
    answers.insert(0,ques+"+"+str(temp_mem))
    query=str(answers)
    # print(query)
    # print(query)
    client = OpenAI(api_key="----------------api-key----------")
    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
            {"role": "system", "content": "You are a statistical and mathematician who experts in telling complex things in very simple like carl sagan but in conversation manner in brief just like chatting with friend "},
            {"role": "user", "content": query}
                ]
            )
    output=str(completion.choices[0].message.content)
    # memory = ConversationBufferMemory(return_messages=True)
    # memory.save_context({"input": query}, {"output": output})
    # memory.load_memory_variables({})
    # print(memory)
    # ques=str(input("Enter query"))
    #print(f"Your AI friend:{output}")
    memory = ConversationBufferWindowMemory( k=15, return_messages=True)
    memory.save_context({"input": query}, {"output": output})
    temp_mem = memory.load_memory_variables({})
    print(f"AI: {output}")
    #print(temp_mem)