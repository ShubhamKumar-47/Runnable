from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 2. Model
model = ChatMistralAI(model="mistral-small-latest")

# 3. Output Parser
parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke({"topic": "Machine Learning"})
print(result)