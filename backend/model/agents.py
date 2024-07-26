from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

# Initialize the language model
model = Ollama(model = "llama3")

# Placeholder for menu  that needs to be taken from PDF
menu = "Extracted text from the menu PDF"

# Define the agents
menu_parser = Agent(
    role = "menu_parser",
    goal = "accurately parse the menu and extract food items and their descriptions",
    backstory = "You are an AI assistant trained to analyze menu content and identify food items along with their descriptions. You are expected to parse the menu text and extract relevant information.",
    verbose = True,
    allow_delegation = False,
    llm = model
)

recommendation_agent = Agent(
    role = "food_recommender",
    goal = "recommend food items based on extracted menu items and user preferences",
    backstory = "You are an AI assistant trained to recommend food items based on menu content and user preferences. You should provide a list of suggested dishes that match the user's tastes.",
    verbose = True,
    allow_delegation = False,
    llm = model
)

# Define the tasks
parse_menu = Task(
    description = "Parse the menu and extract food items and their descriptions.",
    agent = menu_parser,
    expected_output = "A list of food items with descriptions."
)

recommend_food = Task(
    description = "Recommend food items based on user preferences and the parsed menu.",
    agent = recommendation_agent,
    expected_output = "A list of recommended dishes based on the user's tastes."
)

# Define the crew and kick off the process
crew = Crew(
    agents = [menu_parser, recommendation_agent],
    tasks = [parse_menu, recommend_food],
    verbose = 2,
    process = Process.sequential
)

output = crew.kickoff()
print(output)
