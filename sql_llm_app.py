import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_fireworks import ChatFireworks
from langchain_core.messages import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict, List  # Added List here
from dotenv import load_dotenv
import asyncio
import json
import re

# Load environment variables from .env file
load_dotenv()

class GradingResult(BaseModel):
    grade: int = Field(..., ge=1, le=10)
    reason: str = Field(..., max_length=1000)  # Increased max_length

class ModelGrades(BaseModel):
    grades: Dict[str, GradingResult]

class GraderOutput(BaseModel):
    grades: Dict[str, GradingResult]

def parse_grader_output(text: str, model_names: List[str]) -> ModelGrades:
    try:
        # Extract JSON from the text response
        json_str = re.search(r'\{.*\}', text, re.DOTALL)
        if json_str:
            parsed_output = json.loads(json_str.group())
            return ModelGrades(grades={
                model: GradingResult(**parsed_output['grades'].get(model, {'grade': 1, 'reason': "No grade provided"}))
                for model in model_names
            })
    except json.JSONDecodeError:
        st.error("Invalid JSON format in grader output")
    except Exception as e:
        st.error(f"Error parsing grader output: {e}")
    return ModelGrades(grades={
        model: GradingResult(grade=1, reason="Parsing error")
        for model in model_names
    })

# Initialize models
models = {
    "OpenAI": ChatOpenAI(model="gpt-4o-mini"),
    "Anthropic": ChatAnthropic(model_name="claude-3-haiku-20240307"),
    "Google": ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
    "Groq": ChatGroq(model="llama-3.1-70b-versatile", stop_sequences=None),
    "Fireworks": ChatFireworks(
        model="accounts/fireworks/models/mixtral-8x7b-instruct"
    )
}

# Initialize grading models
grading_models = {
    "OpenAI": ChatOpenAI(model="gpt-4o"),
    "Anthropic": ChatAnthropic(model_name="claude-3-5-sonnet-20240620"),
    "Google": ChatGoogleGenerativeAI(model="gemini-1.5-flash")
}

# Streamlit app
st.title("SQL Question Answering and Grading App")

# Create a sidebar for model selection
st.sidebar.title("Model Selection")
selected_models = {}
for model_name in models.keys():
    selected_models[model_name] = st.sidebar.toggle(f"Use {model_name}")

# User input
sql_question = st.text_area("Enter your SQL question:", height=200)

async def generate_answer(model_name, model, question):
    prompt = (
        "Hey look at this problem and answer it in PostgreSQL. Provide only the SQL query "
        "that can be executed directly. Do not include any markdown annotations, code block "
        "delimiters, or exit characters. The query should be ready to run as-is:\n\n"
        f"{question}"
    )
    response = await model.ainvoke([HumanMessage(content=prompt)])
    return model_name, response.content.strip()

async def get_answers(sql_question, selected_models):
    tasks = [
        generate_answer(model_name, models[model_name], sql_question)
        for model_name, is_selected in selected_models.items()
        if is_selected
    ]
    return await asyncio.gather(*tasks)

if st.button("Generate Answers"):
    if sql_question and any(selected_models.values()):
        answers = dict(asyncio.run(get_answers(sql_question, selected_models)))

        # Grade answers
        parser = PydanticOutputParser(pydantic_object=GraderOutput)
        grades = {}
        for grader_name, grader_model in grading_models.items():
            grading_prompt = f"Grade the following SQL answers to this question: {sql_question} {' '.join([f'{model_name} Answer: {answer}' for model_name, answer in answers.items()])} Provide a JSON object with grades for each of the following models: {', '.join(answers.keys())}. Use the following format strictly: {{ \"grades\": {{ \"ModelName1\": {{ \"grade\": X, \"reason\": \"Brief explanation\" }}, \"ModelName2\": {{ \"grade\": Y, \"reason\": \"Brief explanation\" }}, ... }} }} Each grade should be from 1-10. Keep reasons concise, under 500 characters each."

            grading_response = grader_model.invoke([HumanMessage(content=grading_prompt)])
            grades[grader_name] = parse_grader_output(grading_response.content, list(answers.keys()))

        # Create DataFrame
        scores_data = {}
        for model in answers.keys():
            model_scores = {}
            for grader, grade in grades.items():
                model_scores[grader] = grade.grades[model].grade if model in grade.grades else 0
            model_scores['Average'] = sum(model_scores.values()) / len(model_scores)
            scores_data[model] = model_scores

        df_scores = pd.DataFrame.from_dict(scores_data, orient='index')
        
        # Reorder columns to put 'Average' first
        columns = ['Average'] + [col for col in df_scores.columns if col != 'Average']
        df_scores = df_scores[columns]
        
        # Sort by average score and round all scores
        df_scores = df_scores.sort_values('Average', ascending=False)
        df_scores = df_scores.round(2)

        # Display results
        st.subheader("Results")
        st.dataframe(df_scores)

        # Display generated answers in expandable sections
        st.subheader("Generated Answers:")
        for model_name, answer in answers.items():
            with st.expander(f"{model_name} Answer"):
                st.code(answer, language="sql")

        # Display detailed grading in expandable sections
        st.subheader("Detailed Grading:")
        for grader_name, grade in grades.items():
            with st.expander(f"{grader_name} Grading"):
                for model, result in grade.grades.items():
                    st.write(f"**{model}**: Grade {result.grade}/10")
                    st.write(f"Reason: {result.reason}")
                    st.write("---")

        # Add buttons for further actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Results"):
                st.write("Results saved! (Implement actual saving logic)")
        with col2:
            if st.button("Share Results"):
                st.write("Results shared! (Implement actual sharing logic)")

else:
    st.write("Please enter a SQL question and select at least one model.")