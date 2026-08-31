"""Wiring: the three AgentSpecs and the live client.

This is the only module that knows all three roles at once, and it is where the
separation of concerns is *composed*: each spec takes exactly one kit bundle.
Everything else (prompts, loop, dispatchers, graph) is role-agnostic or
role-specific, never both.
"""

from __future__ import annotations

import multi_agent_tools as mat  # the crew package bootstrap put the kit on sys.path

from crew.agent_loop import AgentSpec, OutputMode
from crew.config import OUTPUT_MODE, require_api_key
from crew.prompts import comms_prompt, decision_prompt, researcher_prompt
from crew.schemas import AgentName, CommsOutput, DecisionOutput, ResearcherOutput


def build_specs(output_mode: OutputMode = OUTPUT_MODE) -> dict[AgentName, AgentSpec]:
    return {
        "researcher": AgentSpec(name="researcher", system_prompt=researcher_prompt(output_mode),
                                tools=mat.RESEARCHER_TOOLS, output_model=ResearcherOutput, output_mode=output_mode),
        "decision": AgentSpec(name="decision", system_prompt=decision_prompt(output_mode),
                              tools=mat.DECISION_TOOLS, output_model=DecisionOutput, output_mode=output_mode),
        "comms": AgentSpec(name="comms", system_prompt=comms_prompt(output_mode),
                           tools=mat.COMMS_TOOLS, output_model=CommsOutput, hygiene_field="customer_reply",
                           output_mode=output_mode),
    }


def live_client():
    """The real Anthropic client. Imported lazily so offline tests never need the SDK or a key."""
    from anthropic import Anthropic

    require_api_key()
    return Anthropic()
