from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List

class QueryProcessor:
    def __init__(self, openai_api_key: str = None, llm_model: str = "gpt-4", temperature: float = 0.0):
        if openai_api_key:
            self.llm = ChatOpenAI(model=llm_model, temperature=temperature, openai_api_key=openai_api_key)
        else:
            self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        
        template = """
You are an expert Bengali text analyzer specializing in extracting precise information from literary texts.

CRITICAL RULES:
1. Answer ONLY from the provided context - never use external knowledge
2. For Bengali questions, provide Bengali answers
3. Give the MOST DIRECT answer possible - usually just the name, number, or specific term
4. Study these examples carefully:

EXAMPLES:
Question: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
Answer: শুম্ভুনাথ

Question: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
Answer: মামাকে

Question: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
Answer: ১৫ বছর

ANSWER FORMAT RULES:
- "কাকে বলা হয়েছে?" → Just the name (e.g., "শুম্ভুনাথ")
- "কাকে বলে উল্লেখ করা হয়েছে?" → Name with case ending (e.g., "মামাকে")  
- "কত বছর/কত বয়স?" → Number + "বছর" (e.g., "১৫ বছর")
- "কী/কোন?" → The specific term or object
- "কোথায়?" → The place name
- "কখন?" → The time

If exact answer not found in context, say: "তথ্য পাওয়া যায়নি"

Context:
{context}

Question: {question}

Answer:"""
        
        self.prompt = PromptTemplate.from_template(template)
        self.chat_history = []
    
    def process_query(self, query: str, retrieved_docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        formatted_prompt = self.prompt.format(
            context=context,
            question=query
        )
        
        response = self.llm.invoke(formatted_prompt)
        answer = response.content.strip()
        
        self.chat_history.append({"question": query, "answer": answer})
        
        return answer