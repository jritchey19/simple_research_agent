import argparse
import os

from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

if __name__ == "__main__":

    # Load our .env file
    load_dotenv()

    # Setup parser to except a topic and and optional output flag
    parser = argparse.ArgumentParser(
        prog="Research Agent",
        description="A simple agent that will go out and research a topic"
    )

    parser.add_argument("topic", help="Topic to research about.")
    parser.add_argument("-o", "--output", help="path to output.")

    args = parser.parse_args()

    # Set vars to arguments
    topic = args.topic
    output = args.output

    # Setup our llm to use grok
    grok_llm = LLM(
      model=os.getenv("MODEL"),
      api_key=os.getenv("XAI_API_KEY"),
      temperature=0.7,
      max_tokens=8192,
    )
    
    # This is our agent and how it should do research.
    researcher = Agent(
        role=f"{topic} Senior Researcher",
        goal=f"Search the internet to understand {topic} and report acurate findings",
        backstory=f"You a senior {topic} data researcher who is thourgh when researching a {topic}. You always use two or more sources when citing a fact to ensure accuracy and cite those sources. When you cant you always mark this as a single source and cite it. When you give an opinion you let the user know with [OPINION] before the thought.",
        llm=grok_llm,
        verbose=True
    )

    # The task that we want the Agent to do.
    task = Task(
        description=f"""
        Conduct thorough research about {topic}. Use web search to find current,
        credible information. The current year is 2026.
        """,
        expected_output="""
        a markdown report with clear sections: key trends, notable tools or companies,
        and implications. Aim for 800-1500 words. No fenced code blocks around the whole document.
        """,
        agent=researcher,
    )

    # Setup our crew with the agent and the task to work on.
    crew = Crew(agents=[researcher],tasks=[task])
    
    # Conduct research.
    result = crew.kickoff()

    # If output flag was given and file, write to disk.
    if output:
        with open(output, "w") as f:
            f.write(result.raw)

    # Print the results.
    print(result.raw)
