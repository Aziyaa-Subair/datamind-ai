"""Glue: run the agent, then verify its answer with deterministic code."""

from typing import Any

import pandas as pd

from app.agents.analysis_agent import Deps, build_agent
from app.schemas import ToolCall, VerifiedAnswer
from app.services.dataset_service import profile_dataset
from app.services.verifier import verify


def answer_question(df: pd.DataFrame, question: str, agent: Any = None) -> VerifiedAnswer:
    agent = agent or build_agent()
    deps = Deps(df=df)
    deps.trace.append(
        ToolCall(name="get_dataset_profile", args={}, result=profile_dataset(df))
    )
    run = agent.run_sync(question, deps=deps)
    report = verify(run.output, deps.trace)
    return VerifiedAnswer(result=run.output, report=report, tool_calls=deps.trace)