import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
import json
import anthropic
import openlit

# Load environment variables
load_dotenv(find_dotenv())

openlit.init(otlp_endpoint="http://127.0.0.1:4318", disable_metrics=True)

class MCQ(BaseModel):
    question: str
    options: str
    correct_answer: str


client = anthropic.Anthropic()


def get_latex(query: str):
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0,
        system="""You are a profession latext assistant. change the mathametical question as is into latex. "
               "Dont modify anything. follow the below example while you convert,
            
            examples:
            ------------------
            # The derivative question in LaTeX
            "\text{What is the derivative of the function } f(x) = x^2 \sin(x) \text{ at the point } x = \frac{\pi}{2}\text{?}"
            
            # The integral question in LaTeX
            "\text{Evaluate the integral } I = \int \frac{\sin^3 x}{\cos^2 x} \, dx"
            
            Don't output `Heres the question converted to LaTeX:` etc.

        """,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }
        ]
    )
    print(f"from claud: {message.content[0].text}")
    return message.content[0].text


# Set page config
st.set_page_config(page_title="Competitive Math MCQ Generator", layout="wide")

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .question-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .option-box {
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        margin: 5px 0;
    }
    .correct-answer {
        border-color: #c3e6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Competitive Math MCQ Generator")
st.markdown("Generates high-quality mathematics questions with answer.")

# Input section
topic = st.text_input("Enter the mathematical topic:", "differentiation with trigonometric function")

if st.button("Generate MCQ"):
    with st.spinner("Generating question..."):
        # IIT Mathematics Educator Agent
        mcq_creator = Agent(
            role="IIT Mathematics Educator",
            goal="Always Create a new challenging IIT-JEE level MCQs on mathematics.",
            backstory=(
                "You are a renowned mathematician and educator, known for crafting high-quality IIT-JEE "
                "level mathematics questions. Your questions are designed to test deep conceptual "
                "understanding and problem-solving skills."
            ),
            allow_delegation=False,
            verbose=True,
            memory=False,
            llm=LLM(
                model="claude-3-5-sonnet-20241022")
        )

        # Task to generate IIT-level MCQs
        mcq_task = Task(
            description=(
                "Always Generate a new high-quality complex IIT-JEE level multiple-choice question (MCQ) "
                "on the topic of {topic}. Ensure that the question tests conceptual understanding and problem-solving"
                " ability. Always try to generate a new problem to solve"
                "The output should include:\n"
                "- A clear and concise new question statement.\n"
                "- Four options, with one correct answer and three distractors."
            ),
            expected_output=(
                "An MCQ formatted as follows:\n"
                "- Question: [Question statement]\n"
                "- Options: [Option A, Option B, Option C, Option D]\n"
                "- Correct Answer: [Correct option]\n"
                "No Explanation is needed just the question with options is what is all needed"
            ),
            agent=mcq_creator,
            async_execution=False,
            output_pydantic=MCQ
        )

        # Crew to orchestrate the Agent and Task
        math_mcq_crew = Crew(
            agents=[mcq_creator],
            tasks=[mcq_task],
            process=Process.sequential
        )

        # Get the result
        result = math_mcq_crew.kickoff(inputs={'topic': topic}).pydantic.model_dump_json()
        print(f"result: {result}")
        result = json.loads(result)
        # Parse the result
        print(f"question: {result['question']}")
        print(f"options: {result['options']}")

        question = get_latex(result["question"])
        options = result["options"].split(",")
        correct_answer = result["correct_answer"]

        # Display the question
        st.markdown("### Question")
        st.latex(question)

        # Display options
        st.markdown("### Options")
        for i, option in enumerate(options):
            is_correct = option.strip().startswith(correct_answer.split(')')[0])
            box_class = 'option-box correct-answer' if is_correct else 'option-box'
            st.markdown(f"<div class='{box_class}'>{option}</div>", unsafe_allow_html=True)

        # Display correct answer
        st.markdown("### Correct Answer")
        st.markdown(f"<div class='question-box'>{correct_answer}</div>", unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using CrewAI and Streamlit")
