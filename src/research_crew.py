from crewai import Agent, Crew, Process, Task

class ResearchCrew:

    def researcher(self, agent_role, agent_goal, agent_backstory, useLLM, verb=False):
        self.researcher = Agent(
            role=agent_role,
            goal=agent_goal,
            backstory=agent_backstory,
            llm=useLLM,
            verbose=verb,
        )

    def writer(self, writer_role, writer_goal, writer_backstory, useLLM, verb=False):
        self.writer = Agent(
            role=writer_role,
            goal=writer_goal,
            backstory=writer_backstory,
            llm=useLLM,
            verbose=verb,
        )

    def editor(self, useLLM, verb=False):
        self.editor = Agent(
            role="Senior Publisher Editor",
            goal="Review documentation using industry standard for publishing and editing with a blend of objective and structural insight, meticulous attention to detail (grammer/consistency), focusing on strong communication and deep knowledge of genere and reader expectations.",
            backstory="You have worked in the publishing industry for 20+ years, award winning and highly sought after for your skills as an editor. You know what readers expect from technical documents, primers, and textbooks. You help enhance the manuscript you are editing while helping to develop the writer unique voice.",
            llm=useLLM,
            verbose=verb,
        )

    def task(self, t1_description, t1_expected_output, t2_description, t2_expected_output):
        self.task1 = Task(
            description = t1_description,
            expected_output = t1_expected_output,
            agent = self.researcher,
            process = Process.sequential
        )
        self.task2 = Task(
            description = t2_description,
            expected_output = t2_expected_output,
            agent = self.writer,
            process = Process.sequential
        )
        self.task3 = Task(
            description = "Review and edit the writing of the previous agent, ensuring that the documentation is structure well for its genere. Consider the working and grammatical form of the writing. Ensure the title corrilates to the subject of the writing.",
            expected_output = "A markdown file that is polished in its writing and design as a technical writing (textbook, primer, paper, etc)",
            agent = self.editor,
            process = Process.sequential
        )

    def crew(self):
        self.crew = Crew(agents=[self.researcher, self.writer, self.editor],tasks=[self.task1, self.task2, self.task3])

    def get_results(self):
        return self.crew.kickoff()
