#!/usr/bin/env python
from pathlib import Path
from pyprojroot import here

from crewai.flow import Flow, listen, start

import sys

sys.path.insert(0, str(here()))      # project root — makes toolbox/, Utils/ importable
sys.path.insert(0, str(here("src")))  # src root

from etl_agent.crews.cleaning_agent.content_crew import ContentCrew


class ContentFlow(Flow):
    """Two-step ETL flow: clean → model."""

    @start()
    def run_cleaning_crew(self):
        """Step 1: run the cleaning + modelling crew (sequential tasks)."""
        result = ContentCrew().crew().kickoff()
        print("Cleaning & Modelling result:", result)
        return result

    @listen(run_cleaning_crew)
    def on_crew_complete(self, crew_result):
        """Step 2: post-process or pass-through the crew result."""
        print("Flow complete. Crew output:", crew_result)
        return crew_result


def kickoff():
    flow = ContentFlow()
    flow.plot()
    flow.kickoff()


if __name__ == "__main__":
    kickoff()
