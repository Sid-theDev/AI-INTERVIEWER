import streamlit as st
from typing import Literal
from dataclasses import dataclass
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.llms import Replicate
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import nltk
from prompts.prompts import templates
from langchain.chat_models import ChatOpenAI


@dataclass
class Message:
    """Class for keeping track of interview history."""
    origin: Literal["human", "ai"]
    message: str


def save_vector(text):
    """Embeddings."""
    nltk.download('punkt')
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
    # Creating embeddings
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch


def initialize_session_state_jd():
    """Initialize session state for job description interview."""
    if 'jd_docsearch' not in st.session_state:
        st.session_state.jd_docsearch = save_vector(jd)
    if 'jd_retriever' not in st.session_state:
        st.session_state.jd_retriever = st.session_state.jd_docsearch.as_retriever(
            search_type='similarity')
    if 'jd_chain_type_kwargs' not in st.session_state:
        Interview_prompt = PromptTemplate(
            input_variables=["context", "question"], template=templates.jd_template)
        st.session_state.jd_chain_type_kwargs = {"prompt": Interview_prompt}
    if 'jd_memory' not in st.session_state:
        st.session_state.jd_memory = ConversationBufferMemory()
    if "jd_history" not in st.session_state:
        st.session_state.jd_history = []
        st.session_state.jd_history.append(Message("ai",
                                                   "Hello, Welcome to the interview. I am your interviewer today. I will "
                                                   "ask you professional questions regarding the job description you "
                                                   "submitted. "
                                                   "Please start by introducing a little bit about yourself. Note: The maximum length of your answer is 4097 tokens or approx 1000 words"
                                                   ))
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "jd_guideline" not in st.session_state:
        llm = ChatOpenAI()
        st.session_state.jd_guideline = RetrievalQA.from_chain_type(
            llm=llm, chain_type_kwargs=st.session_state.jd_chain_type_kwargs, chain_type="stuff",
            retriever=st.session_state.jd_retriever, memory=st.session_state.jd_memory).run(
            "Create an interview guideline and prepare only one question for each topic. Make sure the questions test the technical knowledge")
    if "jd_screen" not in st.session_state:
        llm = ChatOpenAI()
        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template="""I want you to act as an interviewer strictly following the guideline in the current conversation.
                               Candidate has no idea what the guideline is.
                               Ask me questions and wait for my answers. Do not write explanations.
                               Ask questions like a real person, only one question at a time.
                               Do not ask the same question.
                               Do not repeat the question.
                               Do ask follow-up questions if necessary.
                               Your name is GPTInterviewer.
                               I want you to only reply as an interviewer.
                               Do not write all the conversation at once.
                               If there is an error, point it out.
                               Ask questions such that the total token count does not exceed 4097 tokens.
                               Ask questions that are relevant to the job description. 
                               Ask some real-time coding questions to evaluate his/her knowledge if Job requires it. 
                               The questions must have a combination of simple, medium, and hard level questions.
                               Ask as many questions as needed to really evaluate the candidate's knowledge,skills and attitude.
                               Current Conversation:
                               {history}
                               Candidate: {input}
                               AI: """)
        st.session_state.jd_screen = ConversationChain(
            prompt=PROMPT, llm=llm, memory=st.session_state.jd_memory)
    if 'jd_feedback' not in st.session_state:
        llm = ChatOpenAI()
        st.session_state.jd_feedback = ConversationChain(
            prompt=PromptTemplate(
                input_variables=["history", "input"], template=templates.feedback_template),
            llm=llm,
            memory=st.session_state.jd_memory,
        )


def answer_call_back():
    """Handle user's answer callback."""
    with get_openai_callback() as cb:
        # user input
        human_answer = st.session_state.answer
        user_input = human_answer  
        st.session_state.jd_history.append(
            Message("human", user_input)
        )
        # OpenAI answer and save to history
        llm_answer = st.session_state.jd_screen.run(user_input)
        st.session_state.jd_history.append(
            Message("ai", llm_answer)
        )
        st.session_state.token_count += cb.total_tokens


# Get user input for job description
jd = st.text_area(
    "Please enter the job description here (If you don't have one, enter keywords, such as PostgreSQL or Python instead): "
)

if jd:
    initialize_session_state_jd()
    credit_card_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        feedback = st.button("Get Interview Feedback")
    with col2:
        guidline = st.button("Show me interview guideline!")
    chat_placeholder = st.container()
    answer_placeholder = st.container()

    if guidline:
        st.write(st.session_state.jd_guideline)
    if feedback:
        evaluation = st.session_state.jd_feedback.run(
            "Please give an evaluation regarding the interview")
        st.markdown(evaluation)
        st.download_button(label="Download Interview Feedback",
                           data=evaluation, file_name="interview_feedback.txt")
        st.stop()
    else:
        with answer_placeholder:
            answer = st.text_input("Your Answer")

            if answer:
                st.session_state['answer'] = answer
                answer_call_back()
    with chat_placeholder:
        for answer in st.session_state.jd_history:
            if answer.origin == "ai":
                with st.chat_message("assistant"):
                    st.write(answer.message)
            else:
                with st.chat_message("user"):
                    st.write(answer.message)

    credit_card_placeholder.caption(f"""
    Progress: {int(len(st.session_state.jd_history) / 30 * 100)}% completed.""")

else:
    st.info("Please submit a job description to start the interview.")
