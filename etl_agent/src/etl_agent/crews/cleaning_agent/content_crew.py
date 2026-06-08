from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from etl_agent.tools.custom_tool import (
    normalise_cleaning_tool,
    ask_user,
    pearson_tool,
    describe_dataset_tool,
)
from dotenv import load_dotenv
import os
from pyprojroot import here

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
load_dotenv(here(".env"))

api = os.getenv("DEEPSEEK_API_KEY")


@CrewBase
class ContentCrew:
    """Content Crew — sequential cleaning → modelling pipeline"""

    agents: list[BaseAgent]
    tasks: list[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would like to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    @staticmethod
    def _llm() -> LLM:
        """Shared LLM factory — all agents use the same DeepSeek configuration."""
        return LLM(
            model="deepseek-chat",
            api_key=api,
            base_url="https://api.deepseek.com/v1",
            temperature=0.7,
        )

    """
    @agent
    def first_cleaner(self) -> Agent:
        return Agent(
            config=self.agents_config["first_cleaner"],
            llm=self._llm(),
            max_iter=3,
            tools=[normalise_cleaning_tool()],
        )
    """

    @agent
    def data_modelling_a(self) -> Agent:
        return Agent(
            config=self.agents_config["data_modelling_a"],
            llm=self._llm(),
            max_iter=3,
            tools=[pearson_tool(), describe_dataset_tool()],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    """
    @task
    def data_cleaning(self) -> Task:
        return Task(
            config=self.tasks_config["data_cleaner"],  # type: ignore[index]
            human_input=True,
        )
    """

    @task
    def data_modelling_t(self) -> Task:
        return Task(
            config=self.tasks_config["data_modelling"],  # type: ignore[index]
            human_input=True, # modelling receives cleaning output
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Content Crew — sequential execution (no manager needed)."""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically populated by @agent decorators
            tasks=self.tasks,    # Automatically populated by @task decorators
            process=Process.hierarchical,
            manager_llm= self._llm(),
            verbose=True,
        )
