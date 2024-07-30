# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI
#to fetch data from url
loader = WebBaseLoader('https://en.wikipedia.org/wiki/Large_language_model')
#data is stored in variable
data = loader.load()
#divide large data into smaller parts call chunk
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#splited data is then stored in variable
docs = text_splitter.split_documents(data)
# print(docs)
#use openai embeddings to convert text data into a vector with having dimension and vector index form for vector database
embeddings = OpenAIEmbeddings(api_key="------api key-----")
#embedded vector database is then stored in faiss(facebook AI similarity search)
db = FAISS.from_documents(docs, embeddings)
# print(db.index.ntotal)
#query asked by client
ques="give me the list of names of algorithms and it's techniques for NLP "
#method in faiss to find similar data relate to the query
answers = db.similarity_search(ques)
# print(type(answers))
#insert the query in first index
answers.insert(0,ques)
#list data convert into a string form
query=str(answers)
# print(query)
client = OpenAI(api_key="-----api_key--------")
completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
            {"role": "system", "content": "You are a statistical and mathematician who experts in telling complex things in very simple like carl sagan"},
            {"role": "user", "content": query}
                ]
            )
print(completion.choices[0].message.content)