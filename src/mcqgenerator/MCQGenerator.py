import os
from dotenv import load_dotenv

from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=OPENAI_API_KEY
)

# Step 1: Prompt to generate MCQs
quiz_generation_prompt = PromptTemplate.from_template("""
Text:
{text}

You are an expert MCQ Maker. Given the above text, your job is to \
create a quiz of {number} multiple choice questions for {subject} students in a {tone} tone.

- Ensure the questions are **not repeated**.
- Ensure each question is supported by the input text.
- Follow this exact structure for the output:
### RESPONSE_JSON
{response_json}

Return only the JSON content in the format shown above.
""")

quiz_generation_chain = quiz_generation_prompt | llm | StrOutputParser()

# Step 2: Prompt to evaluate the quiz
quiz_evaluation_prompt = PromptTemplate.from_template("""
You are an expert English grammarian and educator. Given a Multiple Choice Quiz designed for {subject} students:

1. Evaluate the **complexity and appropriateness** of each question.
2. Provide a short **50-word analysis** of the quiz.
3. If needed, **modify questions or tone** to better match the student level.

Quiz:
{quiz}

Provide your expert review below:
""")

quiz_evaluation_chain = quiz_evaluation_prompt | llm | StrOutputParser()

# Combined sequential chain using RunnableSequence
generate_evaluate_chain = RunnableSequence(
    steps=[
        {
            "quiz": quiz_generation_chain
        },
        {
            "review": lambda x: quiz_evaluation_chain.invoke({
                "subject": x["subject"],
                "quiz": x["quiz"]
            }),
            "quiz": lambda x: x["quiz"]
        }
    ]
)
