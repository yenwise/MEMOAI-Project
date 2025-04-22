"""
category_classifier.py

This module provides text classification functionality for MemoAI.
It uses a language model or predefined logic to categorize text notes into
useful groups such as reminders, tasks, or ideas.
"""
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("...")

# OpenAI LLM Model
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# Prompt Template
category_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Categorize the following note in the best way. Available categories:
    - 'Calendar': Things related to time, date, or appointments.
    - 'Reminder': Notes intended to remind about something.
    - 'Movie': Movie or TV show suggestions.
    - 'Other': Everything else.

    Note: {text}
    Return only the category name as the answer.
    """
)

def categorize_text(text):
    """
    Analyze the input text and determine its appropriate category.
    """
    try:
        formatted_prompt = category_prompt.format(text=text)
        response = llm.invoke(formatted_prompt)
        return response.content.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

if __name__ == "__main__":
    test_note = "Watch the Tetris movie"
    category = categorize_text(test_note)
    print(f"📝 Note: {test_note}\n🏷️ Category: {category}")

