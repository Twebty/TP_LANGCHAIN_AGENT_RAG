import re
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    dynamic_prompt,
    ModelRequest,
    ModelResponse,
    HumanInTheLoopMiddleware,
)
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

load_dotenv()


# =========================
# MEMOIRE SIMPLE
# =========================
class SimpleMemory:
    def __init__(self):
        self.history = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_context(self) -> str:
        if not self.history:
            return "Aucun historique."
        return "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in self.history[-8:]
        )


memory = SimpleMemory()


# =========================
# TOOLS PERSONNALISES
# =========================
@tool
def add_numbers(a: float, b: float) -> float:
    """Additionne deux nombres."""
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiplie deux nombres."""
    return a * b


@tool
def save_note(note: str) -> str:
    """Sauvegarde une note dans la mémoire."""
    memory.add("note", note)
    return f"Note sauvegardée : {note}"


@tool
def get_memory() -> str:
    """Retourne la mémoire de session."""
    return memory.get_context()


# =========================
# TOOLS PREDEFINIS
# =========================
duckduckgo_tool = DuckDuckGoSearchRun(name="duckduckgo_search")
tavily_tool = TavilySearchResults(max_results=3, name="tavily_search")
python_repl_tool = PythonREPLTool()


# =========================
# MODELES
# =========================
basic_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
advanced_model = ChatOpenAI(model="gpt-4.1", temperature=0)


# =========================
# MIDDLEWARE : DYNAMIC MODEL
# =========================
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    user_text = ""
    for msg in request.state["messages"]:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_text += msg.get("content", "") + " "

    keywords = ["analyse", "code", "python", "rag", "architecture", "recherche"]

    if len(user_text) > 200 or any(k in user_text.lower() for k in keywords):
        return handler(request.override(model=advanced_model))
    return handler(request.override(model=basic_model))


# =========================
# MIDDLEWARE : DYNAMIC PROMPT
# =========================
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    return f"""
Tu es un assistant pédagogique expert en LangChain.
Réponds clairement en français.
Utilise les tools si nécessaire.
Voici la mémoire récente :
{memory.get_context()}
""".strip()


# =========================
# GARDRails SIMPLE
# =========================
def contains_sensitive_data(text: str) -> bool:
    patterns = [
        r"sk-proj-[A-Za-z0-9_\-]+",
        r"api[\s_-]?key",
        r"password",
        r"mot de passe",
    ]
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


@tool
def guarded_user_echo(text: str) -> str:
    """Retourne le texte si aucun contenu sensible n'est détecté."""
    if contains_sensitive_data(text):
        return "Blocage GuardRails : contenu sensible détecté."
    return f"Texte autorisé : {text}"


# =========================
# HUMAN IN THE LOOP
# =========================
hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "python_repl": True,
        "tavily_search": True,
    }
)


# =========================
# AGENT
# =========================
agent = create_agent(
    model=basic_model,
    tools=[
        add_numbers,
        multiply_numbers,
        save_note,
        get_memory,
        guarded_user_echo,
        duckduckgo_tool,
        tavily_tool,
        python_repl_tool,
    ],
    middleware=[
        dynamic_model_selection,
        dynamic_system_prompt,
        hitl,
    ],
)


def ask_agent(user_input: str) -> str:
    memory.add("user", user_input)

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    output = result["messages"][-1].content
    memory.add("assistant", output)
    return output


if __name__ == "__main__":
    print("=== PARTIE 1 : AGENT SIMPLE ===")
    while True:
        question = input("\nVous : ")
        if question.lower() in {"quit", "exit"}:
            break
        try:
            answer = ask_agent(question)
            print(f"\nAgent : {answer}")
        except Exception as e:
            print(f"\nErreur : {e}")