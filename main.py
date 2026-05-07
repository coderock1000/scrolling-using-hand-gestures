from hf import generate_response
from groq import generate_response as groq_generate_response
import re
import streamlit as st

def looks_incomplete(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return True
    t = text.strip()
    if t.endswith(("***", "*", "-", "—", ":", ",", "(", "[", "{")):
        return True
    if re.search(r"\d+\.\s*\*\*\s*", t):  
        return True
    if not re.search(r"[.!?]\s*$", t): 
        return True
    return False

def complete_answer(question: str, max_rounds: int = 2) -> str:
    base_prompt = (
        "Answer clearly in numbered points. "
        "Do not cut sentences. Finish each point fully.\n\n"
        f"Question: {question}"
    )

    ans = generate_response(base_prompt, temperature=0.3, max_tokens=1024)

    rounds = 0
    while rounds < max_rounds and looks_incomplete(ans):
        cont_prompt = (
            "Continue EXACTLY from where you stopped. "
            "Do NOT repeat earlier text. "
            "Finish the incomplete point and complete the answer.\n\n"
            f"Question: {question}\n\n"
            f"Answer so far:\n{ans}\n\nContinue:"
            )
        
        more = generate_response(cont_prompt, temperature=0.3, max_tokens=1024)
        ans = (ans.rstrip() + "\n" + more.lstrip()).strip()
        rounds += 1

    return ans

def main():
    st.title("AI Teaching Assistant")
    st.write("Welcome! You can ask me anything about various subjects, and I'll provide an answer.")

    user_input = st.text_input("Enter your question here:")

    if user_input:
        st.write(f"**Your question:** {user_input}")
        response = complete_answer(user_input)
        st.write("**AI's answer:**")
        st.markdown(response)  # markdown renders numbered points nicely
    else:
        st.info("Please enter a question to ask.")

if __name__ == "__main__":
    main()
