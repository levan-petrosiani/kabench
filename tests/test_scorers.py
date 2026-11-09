from kabench import scorers
from kabench.tracks.agent import score as agent_score
from kabench.tracks.agent_world import World


def test_normalize_strips_wrappers():
    assert scorers.normalize('Output: "მან თქვა, რომ მოვა."') == "მან თქვა, რომ მოვა."


def test_edit_scores_over_correction():
    s = scorers.edit_scores("მე შენ გხედავ.", "მეს შენს გხედავ.", "მე შენ გხედავ.")
    assert s["over_correction"] and not s["needs_fix"] and s["gain"] == -1.0


def test_edit_scores_missed_and_gain():
    missed = scorers.edit_scores("ხუთი ბავშვები მოვიდნენ.", "ხუთი ბავშვები მოვიდნენ.", "ხუთი ბავშვი მოვიდა.")
    assert missed["missed"] and missed["gain"] == 0
    fixed = scorers.edit_scores("ხუთი ბავშვები მოვიდნენ.", "ხუთი ბავშვი მოვიდა.", "ხუთი ბავშვი მოვიდა.")
    assert fixed["gain"] == 1.0


def test_format_ok():
    assert scorers.format_ok("მან თქვა, რომ მოვა.", "მან თქვა რომ მოვა.")
    assert not scorers.format_ok("მან თქვა, რომ მოვა.\n\nახსნა: ...", "მან თქვა რომ მოვა.")
    assert not scorers.format_ok("**მან თქვა, რომ მოვა.**", "მან თქვა რომ მოვა.")


def test_georgian_ratio_and_language_match():
    assert scorers.georgian_ratio("გამარჯობა") == 1.0
    assert scorers.georgian_ratio("hello 16°C") == 0.0
    assert scorers.language_match("ბათუმში 16°C, cloudy", "ka")
    assert not scorers.language_match("It is cloudy in Batumi.", "ka")


def test_parse_choice():
    assert scorers.parse_choice("2) პასუხი") == 2
    assert scorers.parse_choice("Answer: 4") == 4
    assert scorers.parse_choice("none") is None


def test_contact_search_handles_georgian_case_suffix():
    w = World("ka")
    assert [r["id"] for r in w.search_contacts("გიორგის")["results"]] == ["c3"]
    assert len(w.search_contacts("ნინოს")["results"]) == 2
    assert w.search_contacts("Giorgi")["results"][0]["name"] == "გიორგი მაისურაძე"


def _out(calls, text=""):
    return {"text": text, "calls": calls, "steps": 1, "invalid_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def test_agent_score_calls_and_args():
    exp = {"calls": [{"tool": "search_contacts"},
                     {"tool": "send_message", "args": {"contact_id": ["c3"], "text": ["10"]}}]}
    calls = [{"tool": "search_contacts", "args": {"query": "გიორგი"}},
             {"tool": "send_message", "args": {"contact_id": "c3", "text": "10 წუთით დავაგვიანებ"}}]
    s = agent_score(output=_out(calls, "გავუგზავნე."), expected_output=exp, metadata={"lang": "ka"})
    assert s["success"] and s["language_match"]
    bad = [{"tool": "send_message", "args": {"contact_id": "Giorgi", "text": "10 min"}}]
    s = agent_score(output=_out(bad), expected_output=exp, metadata={"lang": "en"})
    assert not s["success"] and not s["tools_ok"]


def test_agent_score_side_effects_fail_clarify():
    exp = {"clarify": True, "no_side_effects": True}
    calls = [{"tool": "create_calendar_event", "args": {"title": "x", "date": "2026-11-08", "time": "10:00"}}]
    s = agent_score(output=_out(calls, "რომელ ნინოს?"), expected_output=exp, metadata={"lang": "ka"})
    assert not s["success"] and s["clarify_ok"] and not s["no_side_effects_ok"]


def test_language_specific_arg_expectations():
    exp = {"calls": [{"tool": "create_calendar_event",
                      "args": {"title": {"en": ["project review"], "ka": ["პროექტის მიმოხილვა"]}}}]}
    calls = [{"tool": "create_calendar_event", "args": {"title": "Project review", "date": "d", "time": "t"}}]
    assert not agent_score(output=_out(calls), expected_output=exp, metadata={"lang": "ka"})["args_ok"]


def test_speed_guard_trips_only_on_consecutive_slow_items():
    from kabench.runner import SpeedGuard

    g = SpeedGuard(min_tps=3, max_latency_s=300, strikes=3)
    slow, fast = {"latency_s": 20, "completion_tokens": 40}, {"latency_s": 2, "completion_tokens": 40}
    for out in (slow, slow, fast, slow, slow):
        g.check(out)
    assert not g.tripped  # the fast item reset the streak
    g.check(slow)
    assert g.tripped


def test_speed_guard_ignores_short_outputs_and_errors():
    from kabench.runner import SpeedGuard

    g = SpeedGuard(strikes=2)
    for _ in range(5):
        g.check({"latency_s": 8, "completion_tokens": 1})  # long prompt, 1-token answer: fine
        g.check({"error": "boom"})
    assert not g.tripped
    g.check({"latency_s": 400, "completion_tokens": 1})
    g.check({"latency_s": 400, "completion_tokens": 1})
    assert g.tripped


def test_speed_guard_allows_long_fast_reasoning():
    from kabench.runner import SpeedGuard

    g = SpeedGuard(strikes=2)
    for _ in range(5):
        g.check({"latency_s": 600, "completion_tokens": 6000})  # 10 tok/s: slow-ish but healthy
    assert not g.tripped


def test_grammar_sample_is_nested_and_stable():
    from kabench.tracks.grammar import load_items

    ids = lambda n: [it["id"] for it in load_items("ka", "", sample=n)[0]]
    assert set(ids(10)) <= set(ids(30)) and len(ids(30)) == 30
    assert ids(30) == ids(30)


def test_belebele_sample_is_nested():
    from kabench.belebele import sample

    small, big = [p["id"] for p in sample(30)], [p["id"] for p in sample(100)]
    assert small == big[:30]


def test_speed_guard_trips_on_dead_server_but_not_agent_failures():
    from kabench.runner import SpeedGuard

    g = SpeedGuard(strikes=3)
    for _ in range(5):
        g.check({"error": "no final answer after 8 steps"})
    assert not g.tripped
    for _ in range(3):
        g.check({"error": "APITimeoutError: Request timed out."})
    assert g.tripped


def test_text_tool_calls_and_wrapped_args():
    from kabench.tracks.agent import text_tool_calls
    from kabench.tracks.agent_world import parse_args

    granite = text_tool_calls('<tool_call>[{"arguments": {"city": "Tbilisi"}, "name": "get_weather"}]')
    assert [c["function"]["name"] for c in granite] == ["get_weather"]
    assert parse_args(granite[0]["function"]["arguments"]) == {"city": "Tbilisi"}
    llama = text_tool_calls('{"name": "convert_currency", "parameters": {"amount": 250}}')
    assert parse_args(llama[0]["function"]["arguments"]) == {"amount": 250}
    assert text_tool_calls("<tool_call>[]") == []
    assert text_tool_calls('Here is the call: {"name": "get_weather"}') == []
    assert text_tool_calls("Batumi is warmer today.") == []
    # Aya wraps arguments; ordinary args that happen to contain "parameters" stay as they are
    assert parse_args('{"tool_name": "get_weather", "parameters": {"city": "Gori"}}') == {"city": "Gori"}
    assert parse_args('{"city": "Gori", "parameters": {}}') == {"city": "Gori", "parameters": {}}


def test_wilson_interval():
    from kabench.report import wilson

    lo, hi = wilson(9, 15)
    assert 0.35 < lo < 0.37 and 0.80 < hi < 0.82
    assert wilson(0, 15)[0] == 0.0 and wilson(15, 15)[1] == 1.0
