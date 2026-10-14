from protonous.client import OpenrouterConfig


def test_to_params_omits_none_keeps_model():
    assert OpenrouterConfig(model="x").to_params() == {"model": "x"}


def test_to_params_includes_set_fields():
    p = OpenrouterConfig(model="x", temperature=0.7, top_p=0.9, presence_penalty=0.1).to_params()
    assert p["temperature"] == 0.7 and p["top_p"] == 0.9 and p["presence_penalty"] == 0.1
    assert "max_tokens" not in p


def test_to_params_reasoning_effort_and_extra_body_merge():
    p = OpenrouterConfig(
        model="x", reasoning_effort="minimal", extra_body={"provider": {"data_collection": "deny"}}
    ).to_params()
    assert p["extra_body"]["reasoning"]["effort"] == "minimal"
    assert p["extra_body"]["provider"]["data_collection"] == "deny"


def test_to_params_tool_choice_passthrough():
    assert OpenrouterConfig(model="x", tool_choice="required").to_params()["tool_choice"] == "required"
    forced = {"type": "function", "function": {"name": "f"}}
    assert OpenrouterConfig(model="x", tool_choice=forced).to_params()["tool_choice"] == forced
    assert "tool_choice" not in OpenrouterConfig(model="x").to_params()
