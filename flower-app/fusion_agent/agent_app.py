"""The real Flower AgentApp entry point (Flower 1.37 runtime)."""

import json
import os

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import ConfigRecord, Context
from openai import OpenAI

from .orchestrator import history, investigate
from .sites import load_sites
from .steward import FacilitySteward
from .tools import Toolbox

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    prompt = context.run_config.get("agent.input")
    model = context.run_config.get("agent.model")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
        raise ValueError("agent.input must contain 1-12000 characters")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("agent.model must be configured")
    max_calls = context.run_config.get("agent.max-tool-calls", 16)
    if type(max_calls) is not int or not 1 <= max_calls <= 24:
        raise ValueError("agent.max-tool-calls must be between 1 and 24")
    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
        timeout=60,
    )
    toolbox = Toolbox(
        load_sites(os.environ.get("FUSION_SITES_CONFIG")),
        max_calls,
        steward=FacilitySteward(client, model),
    )
    messages = history(agent.events.get_trace(), context.run_id, prompt.strip())
    try:
        answer = investigate(
            client,
            agent.events,
            toolbox,
            model,
            messages,
            context.run_config.get("agent.max-tool-turns", 8),
        )
        print(answer)
    finally:
        # Only approved tool outputs enter the shared run state/log. No local
        # paths, credentials, or raw trajectories are exported by local adapters.
        audit = json.dumps(toolbox.audit, allow_nan=False)
        context.state["fusion.last_investigation"] = ConfigRecord(
            {"run_id": str(context.run_id), "tool_calls": toolbox.used, "audit": audit}
        )
        print("FUSION_TOOL_AUDIT " + audit)
