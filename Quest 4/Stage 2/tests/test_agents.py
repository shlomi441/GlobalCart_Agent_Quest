"""Phase 4: the wiring pins each role to exactly its bundle, and the prompts name every lock the agent can meet."""

import crew  # noqa: F401
import multi_agent_tools as mat

from crew.agents import build_specs
from crew.prompts import PROMPTS
from crew.run import audit_view
from crew.schemas import CommsOutput, DecisionOutput, ResearcherOutput

LOCKS = {
    "researcher": ["ORDER_NOT_FOUND", "USER_ORDER_MISMATCH", "ID_PROBING_BLOCKED", "REPEATED_CALL"],
    "decision": ["BLOCKED_BY_RISK_REPORT", "BLOCKED_BY_POLICY_ESCALATION", "BLOCKED_BY_POLICY_VERDICT",
                 "DUPLICATE_CLAIM", "NO_REFUND_REQUESTED", "SEQUENCING_VIOLATION", "AMOUNT_NOT_MERITED", "WRONG_ORDER"],
    "comms": ["ARGUMENT_MISMATCH", "PAYLOAD_MISMATCH", "ROUTE_MISMATCH", "ALERT_NOT_AUTHORIZED",
              "ROUTING_NOT_APPLICABLE", "DUPLICATE_ALERT"],
}


def test_each_spec_gets_exactly_its_bundle():
    specs = build_specs("text_json")
    assert [t["name"] for t in specs["researcher"].tools] == [t["name"] for t in mat.RESEARCHER_TOOLS]
    assert [t["name"] for t in specs["decision"].tools] == [t["name"] for t in mat.DECISION_TOOLS]
    assert [t["name"] for t in specs["comms"].tools] == [t["name"] for t in mat.COMMS_TOOLS]
    assert (specs["researcher"].output_model, specs["decision"].output_model, specs["comms"].output_model) == \
        (ResearcherOutput, DecisionOutput, CommsOutput)
    assert specs["comms"].hygiene_field == "customer_reply" and specs["decision"].hygiene_field is None
    assert all(s.output_mode == "tool" for s in build_specs("tool").values())


def test_prompts_name_every_lock_and_the_output_keys():
    for agent, build in PROMPTS.items():
        text = build("text_json")
        for code in LOCKS[agent]:
            assert code in text, f"{agent} prompt does not explain {code}"
        assert "ONLY one JSON object" in text and f"finish_{agent}" not in text
        assert f"finish_{agent}" in build("tool")
    for key in ("ticket_facts", "identity_check", "findings", "tools_called"):
        assert f'"{key}"' in PROMPTS["researcher"]("text_json")
    for key in ("decision", "rationale", "cited_policies"):
        assert f'"{key}"' in PROMPTS["decision"]("text_json")
    assert '"customer_reply"' in PROMPTS["comms"]("text_json")


def test_comms_prompt_lists_the_kits_alert_keys_per_channel():
    text = PROMPTS["comms"]("text_json")
    for ch in mat.get_escalation_channels()["channels"]:
        assert ch["channel_id"] in text and ch["name"] in text
    assert "escalation_reason" in text and "auto_refund_cap" in text and "order_date" in text
    assert '"n/a"' in text                                        # incident 5: no audit -> no score


def test_prompts_stay_small_enough_for_a_weak_model():
    for agent, build in PROMPTS.items():
        assert len(build("text_json")) < 7000, f"{agent} prompt is getting long: {len(build('text_json'))} chars"


def test_audit_view_renders_a_result(tmp_path, isolated_outbox):
    from crew.graph import Crew
    from crew.memory import Ledger
    from fake_client import FakeClient, final
    from test_graph_offline import SPECS, research, research_final, check, decision_final, ROUTE, B1_ALERT, comms_final
    turns = [research("ORD-1005", "USR-105"), research_final("ORD-1005", demanded=480.0),
             check("ORD-1005"), decision_final("ESCALATED_TO_HUMAN", ["POL-ESC-01"]),
             ROUTE, B1_ALERT, comms_final(tools=["get_escalation_route", "send_slack_alert"])]
    result = Crew(SPECS, FakeClient(turns), ledger=Ledger(tmp_path / "l.jsonl")).run("ticket", run_id="demo")
    view = audit_view(result)
    assert "[researcher]" in view and "[decision]" in view and "[comms]" in view
    assert "risk report : complete 90/100 high" in view and "CH-FRAUD critical" in view and "blocked_by=['risk_report:high'" in view
