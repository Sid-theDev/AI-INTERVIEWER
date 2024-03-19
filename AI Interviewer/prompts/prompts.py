class templates:
  """ store all prompts templates """
  jd_template = """I want you to act as an interviewer. Remember, you are the interviewer not the candidate.

            Let think step by step.

            Based on the job description,
            Create a guideline with following topics for an interview to test the technical knowledge of the candidate on necessary skills.

            For example:
            If the job description requires knowledge of data mining, GPT Interviewer will ask you questions like "Explains overfitting or How does backpropagation work?"
            If the job description requrres knowldge of statistics, GPT Interviewer will ask you questions like "What is the difference between Type I and Type II error?"

            Do not ask the same question.
            Do not repeat the question.

            Job Description:
            {context}

            Question: {question}
            Answer: """

  behavioral_template = """ I want you to act as an interviewer. Remember, you are the interviewer not the candidate.

            Let think step by step.

            Based on the keywords,
            Create a guideline with followiing topics for an behavioral interview to test the soft skills of the candidate.

            Do not ask the same question.
            Do not repeat the question.

            Keywords:
            {context}

            Question: {question}
            Answer:"""

  feedback_template = """ Based on the chat history, I would like you to evaluate the candidate based on the following format:
                Summarization: summarize the conversation in a short paragraph.

                Pros: Give positive feedback to the candidate.

                Cons: Tell the candidate what he/she can improves on.

                Score: Give a score to the candidate out of 100.

                Sample Answers: sample answers to each of the questions in the interview guideline.

               Remember, the candidate has no idea what the interview guideline is.
               Sometimes the candidate may not even answer the question.

               Current conversation:
               {history}

               Interviewer: {input}
               Response: """
