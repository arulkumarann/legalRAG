from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.config import config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("legal_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegalDocumentAssistant:
    
    def __init__(self, vectorstore):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            streaming=False,
            temperature=0.3,  
            max_tokens=1024
        )
        
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5,
            input_key="question",   
            output_key="answer"     
        )
        
        self.condense_question_prompt = self._get_condense_question_prompt()
        self.qa_prompt = self._get_qa_prompt()
        
        self.retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4  
            }
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            condense_question_prompt=self.condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True,
            verbose=False
        )
        
        self.conversation_state = {
            "conversation_turn": 0,
            "document_references": set(),
            "legal_terms_explained": set(),
            "unclear_queries": [],
            "current_focus": None
        }
    
    def _get_condense_question_prompt(self):
        return PromptTemplate.from_template(
            """Given the following conversation and a follow-up question, rephrase the follow-up question 
            to be a standalone question that captures the full context of the conversation.
            Focus on legal terminology and document-specific aspects if present.
            
            Chat History:
            {chat_history}
            
            Follow Up Input: {question}
            Standalone question:"""
        )
    
    def _get_qa_prompt(self):
        """Create a prompt for generating answers based on retrieved legal documents."""
        system_template = """You are a knowledgeable and precise legal document assistant. Your role is to help users understand legal documents and answer their questions accurately based on the provided context.

COMMUNICATION GUIDELINES:
- Be clear, precise, and factual in your responses
- Cite specific sections or clauses when appropriate
- Explain legal terminology in plain language
- Do not provide legal advice or interpretation beyond what's in the documents
- Clarify when information is not available in the provided context

RESPONSE FORMAT:
1. Directly answer the user's question based on the documents
2. Include relevant citations to document sections when possible
3. Explain technical legal terms that may be unfamiliar
4. If the question cannot be answered from the documents, clearly state this limitation

Remember to ONLY use the information from the following context. Do not invent details or provide legal advice. 

Context:
{context}"""

        human_template = "{question}"
        
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ]
        
        return ChatPromptTemplate.from_messages(messages)
    
    def _get_clarifying_question(self):
        """Generate an appropriate clarifying question based on conversation state."""
        state = self.conversation_state
        turn = state["conversation_turn"]
        
        if turn <= 2:
            questions = [
                "Which specific section of the document would you like me to focus on?",
                "Is there a particular clause or term you'd like me to explain in more detail?",
                "Would you like me to summarize this section in simpler terms?"
            ]
        elif state["unclear_queries"] and turn <= 4:
            questions = [
                "Could you clarify what you mean by that question?",
                "Which aspect of the document are you most interested in understanding?"
            ]
        elif len(state["document_references"]) >= 2:
            questions = [
                "Would you like me to compare these different sections?",
                "Is there something specific about these clauses that you'd like clarified?"
            ]
        else:
            questions = [
                "Is there anything else about this document you'd like to understand?",
                "Would you like me to explain any other terms or concepts from the document?",
                "Is there a specific implication of this clause you're concerned about?"
            ]
        
        import random
        return random.choice(questions)
    
    def _update_conversation_state(self, query, response):
        """Update conversation state based on the latest exchange."""
        self.conversation_state["conversation_turn"] += 1
        
        reference_indicators = ["section", "clause", "paragraph", "article"]
        for indicator in reference_indicators:
            if indicator in response.lower():
                # Simple extraction of references like "Section 3.2" or "Clause 4"
                import re
                refs = re.findall(r'(?:' + indicator + r')\s+\d+(?:\.\d+)*', response.lower(), re.IGNORECASE)
                for ref in refs:
                    self.conversation_state["document_references"].add(ref)
        
        term_indicators = ["means", "defined as", "refers to", "is a"]
        for indicator in term_indicators:
            if indicator in response.lower():
                import re
                terms = re.findall(r'"([^"]+)"', response)
                for term in terms:
                    if len(term.split()) <= 5:  # Likely a term, not a phrase
                        self.conversation_state["legal_terms_explained"].add(term.lower())
        
        confusion_indicators = ["unclear", "don't understand", "can't find", "not specified"]
        if any(indicator in response.lower() for indicator in confusion_indicators):
            self.conversation_state["unclear_queries"].append(query)
        
        focus_indicators = ["focusing on", "regarding", "about the", "in the"]
        for indicator in focus_indicators:
            if indicator in response.lower():
                import re
                focus = re.findall(r'(?:' + indicator + r')\s+([^.]+)', response.lower())
                if focus:
                    self.conversation_state["current_focus"] = focus[0].strip()
        
        logger.debug(f"Updated conversation state: {self.conversation_state}")
    
    def chat(self, query: str) -> str:
        # Handle exit commands
        if query.lower() in ["exit", "quit", "bye", "goodbye"]:
            return "Thank you for using the Legal Document Assistant. If you have more questions in the future, feel free to ask."
        
        if not query.strip() or len(query.strip()) < 2:
            return "Welcome to the Legal Document Assistant. I'm here to help you understand legal documents. What would you like to know about the documents in my database?"
        
        try:
            logger.info(f"Processing query: {query}")
            
            response = self.chain.invoke({"question": query})
            
            answer = response.get("answer", "").strip()
            logger.debug(f"Raw answer: {answer}")
            
            self._update_conversation_state(query, answer)
            
            source_docs = response.get("source_documents", [])
            if source_docs and "source" not in answer.lower() and "section" not in answer.lower():
                first_doc = source_docs[0]
                if hasattr(first_doc, "metadata") and "source" in first_doc.metadata:
                    answer += f"\n\nSource: {first_doc.metadata['source']}"
            
            if len(answer.split()) > 75 and not answer.endswith("?"):
                clarifying = self._get_clarifying_question()
                answer += f"\n\n{clarifying}"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error during chat: {str(e)}", exc_info=True)
            
            return ("I apologize for the technical issue. As your legal document assistant, "
                   "I'm here to help you understand the documents. Could you please rephrase your question?")
    
    def reset_conversation(self):
        """Reset the conversation state and memory."""
        self.conversation_state = {
            "conversation_turn": 0,
            "document_references": set(),
            "legal_terms_explained": set(),
            "unclear_queries": [],
            "current_focus": None
        }
        
        # Reset memory
        self.memory.clear()
        
        return "Conversation has been reset. How can I help you understand the legal documents today?"