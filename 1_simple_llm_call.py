import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the national animal of Russia?"})
print(result)