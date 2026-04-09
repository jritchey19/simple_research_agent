import argparse
import os

from src.research_crew import ResearchCrew
from crewai import LLM
from dotenv import load_dotenv

if __name__ == "__main__":

    # Load our .env file
    load_dotenv()

    # Setup parser to except a topic and and optional output flag
    parser = argparse.ArgumentParser(
        prog="Research Agent",
        description="A simple agent that will go out and research a topic"
    )

    parser.add_argument("-k", "--keyword", help="Keyword for agent role.", required=True)
    parser.add_argument("-s", "--subject", help="Subject to research about.", required=True)
    parser.add_argument("-o", "--output", help="path to output.")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Set the temperature of the model")
    parser.add_argument("-m", "--max_tokens", type=int, default=8192, help="Sets the maximum number of tokens available to the model")

    args = parser.parse_args()

    # Set vars to arguments
    KEYWORD = args.keyword
    SUBJECT = args.subject
    TEMPERATURE = args.temperature
    MAX_TOKENS = args.max_tokens
    OUTPUT = args.output

    # Setup our llm to use grok
    grok_llm = LLM(
      model=os.getenv("MODEL"),
      api_key=os.getenv("XAI_API_KEY"),
      temperature=TEMPERATURE,
      max_tokens=MAX_TOKENS,
    )
    myResearchCrew = ResearchCrew()
    r_role = f"Academic Research Specialist in {KEYWORD}."
    r_goal = "Discover and synthesize cutting-edge research, identifying key trends, methodologies, and findings while evaluating the quality and reliability of sources by verify with two or more sources and citing them accurately. Ensure to be descriptive and fully fleshout the topic with details like,'How does the topic work, what are known issues, who came up with the topic, are there problems the topic solves, what solutions does this topic have.'"
    r_backstory = f"You are a 20+ year veteran researcher in the academic field who has recently shifted focus on {KEYWORD}. You have strived to be a subject-matter-expert on this topic and have put your arts of digital research that you have master to this task. You've worked with research teams at prestigious universities and know how to navigate academic databases, evaluate research quality, and synthesize findings across disciplines. You're methodical in your approach, always cross-referencing information and tracing claims to primary sources before drawing conclusions."
    myResearchCrew.researcher(r_role, r_goal, r_backstory, grok_llm, verb=True)
    
    w_role = "Technical Writer"
    w_goal = "Write professional documentation that is clear, concise, descriptive and user-friendly with mermaid diagrams that explain complex information simply."
    w_backstory = "As a seasoned writer who has written for some of the biggest Academic journals, you strive to make good documentation on the research given to you. You keep the document on track and truthful while making hard-to-digest information palitable and understandable. You prefer clean documentation that flows easily from one paragraph to the next while describing the subject in full to the reader."
    myResearchCrew.writer(w_role, w_goal, w_backstory, grok_llm, verb=True)

    myResearchCrew.editor(grok_llm, verb=True)

    t1_desc = SUBJECT
    t1_expected_output = "A comprehensive research document that clearly explains the subject with cites sources, quotes from those source with line numbers or paragraphs for validation, details clearly explained and examples."
    t2_desc = "Analyze the research done by the previous agent and write a document that explains the subject in a technical essay, book, or primer. Write this with the clear intention to teach the subject fully to the reader. Use examples and give practice question and answers for each section and ensure that each section is fully described and developed. Quote the research and cite the sources for those quotes. Make mermaid diagrams where it is needed to convey difficult ideas as well as analogies. Ensure that any diagram that is given is correctly formated an working, free from any syntax errors in the text."
    t2_expected_output = "A markdown text file(primer, essay, textbook) with clear sections: introduction, concepts, examples, a Q&A that is well developed. Aim for a maximum of 25000 words per section. Ensure any mermaid diagrams work and have no errors in them. No fenced code blocks around the whole document."
    myResearchCrew.task(t1_desc, t1_expected_output, t2_desc, t2_expected_output)

    myResearchCrew.crew()
    result = myResearchCrew.get_results()

    # researcher = Agent(
    #     role=f"{KEYWORD} Senior Researcher",
    #     goal=f"Search the internet to understand {SUBJECT} and report acurate findings",
    #     backstory=f"You a senior {KEYWORD} data researcher who is thourgh when researching {SUBJECT}. You always use two or more sources when citing a fact to ensure accuracy and cite those sources. When you cant you always mark this as a single source with [NotVerified] and cite it. When you give an opinion you let the user know with [OPINION] before the thought.",
    #     llm=grok_llm,
    #     verbose=True
    # )

    # # The task that we want the Agent to do.
    # task = Task(
    #     description=f"""
    #     Conduct thorough research about {SUBJECT}. Use web search to find current,
    #     credible information. The current year is 2026.
    #     """,
    #     expected_output="""
    #     a markdown report with clear sections: key trends, notable tools or companies,
    #     and implications. Aim for 800-1500 words. No fenced code blocks around the whole document.
    #     """,
    #     agent=researcher,
    # )

    # # Setup our crew with the agent and the task to work on.
    # crew = Crew(agents=[researcher],tasks=[task])
    # 
    # # Conduct research.
    # result = crew.kickoff()

    # If output flag was given and file, write to disk.
    if OUTPUT:
        with open(OUTPUT, "w") as f:
            f.write(result.raw)

