import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ['LANGCHAIN_PROJECT'] = 'Sequential-Chain-Demo'

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_output_tokens=8192, temperature=0.2)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config={
    'run_name':'Sequential Chain',
    'tags':['llm app','report generation','sumarization'],
    'metadata':{
        'model':'gemini-flash-lite-latest',
        'model_temp':0.2,
        'parser':'StrOutputParser',
    }
}

result = chain.invoke({'topic': 'Reasons of Japan to get nuked in World War 2'}, config=config)

print(result)