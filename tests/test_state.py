from regintel.state import AgentState, new_state


def test_new_state_minimal():
    s = new_state("Does our insider trading policy comply with SEC rules?")
    assert s["query"].startswith("Does our")
    assert s["retrieved"] == []
    assert s["errors"] == []


def test_agentstate_is_dict():
    s: AgentState = new_state("q")
    s["sub_questions"] = ["a", "b"]
    assert s["sub_questions"] == ["a", "b"]
