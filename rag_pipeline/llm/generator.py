from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict

def format_prompt(query: str, chunks: List[Dict]) -> str:
    context = "\n\n".join(
        f"[Section: {c['metadata']['section']}]\n{c['text']}"
        for c in chunks
    )
    return f"""You are a financial analysis assistant. 
Answer the question strictly using ONLY the information from the context provided. 
If the answer requires multiple statements, combine them clearly into a concise summary.

If the answer is partially present, respond with the relevant information and state where the context is incomplete.

If absolutely no relevant information exists, respond exactly with: "The provided document does not contain that information."

Context:
{context}

Question:
{query}

Answer:"""

def generate_answer(query: str, chunks: List[Dict], model_name: str = "gpt-4o", dry_run: bool = False) -> str:
    prompt = format_prompt(query, chunks)

    if dry_run:
        print("DRY RUN - Prompt Only:\n" + prompt)
        return prompt
        
        
    # Make sure this import is from langchain_openai.chat_models, not langchain.chat_models
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    messages = [
        SystemMessage(
            content="You are a financial analyst trained to answer questions based ONLY on provided SEC filings."
        ),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    return response.content