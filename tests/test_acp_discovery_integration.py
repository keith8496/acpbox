"""Exercise discovery and selection over real runner stdio against a fake ACP agent."""
import sys
from pathlib import Path

import pytest
import pytest_asyncio

from acpbox.acp_stdio import AcpInvalidRequestError, AcpRunner, AcpStdioError

FAKE = Path(__file__).with_name("fake_acp_discovery.py")
PROMPT = [{"type": "text", "text": "test"}]


@pytest_asyncio.fixture
async def make_runner(tmp_path):
    runners = []

    async def create(scenario="modern"):
        sent = []
        runner = AcpRunner([sys.executable, str(FAKE)], {"FAKE_ACP_SCENARIO": scenario}, str(tmp_path),
                           message_hook=lambda direction, message: sent.append(message) if direction == "send" else None)
        await runner.start()
        runners.append(runner)
        return runner, sent

    yield create
    for runner in runners:
        await runner.stop()


def calls(sent, method):
    return [message["params"] for message in sent if message.get("method") == method]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario,expected", [("modern", ["alpha", "beta"]),
                                                ("config_only", ["alpha", "beta"]),
                                                ("models_only", ["plan", "build"]),
                                                ("conflicting_models", ["alpha", "beta"]),
                                                ("legacy", ["plan", "build"])])
async def test_discovery_prefers_selectable_config_then_modes(make_runner, scenario, expected):
    runner, _ = await make_runner(scenario)
    assert await runner.get_agent_models() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_selects_model_then_effort_using_updated_options(make_runner, stream):
    runner, sent = await make_runner()
    if stream:
        events = [event async for event in runner.run_turn_stream(PROMPT, model_id="beta", reasoning_effort="xhigh")]
        assert ("text", "hello") in events
        assert ("done", "end_turn") in events
    else:
        text, reason, _ = await runner.run_turn(PROMPT, model_id="beta", reasoning_effort="xhigh")
        assert (text, reason) == ("hello", "end_turn")
    selected = calls(sent, "session/set_config_option")
    assert [(x["configId"], x["value"]) for x in selected] == [("model", "beta"), ("thought_level", "xhigh")]
    methods = [x.get("method") for x in sent]
    assert methods.index("session/prompt") > max(i for i, method in enumerate(methods) if method == "session/set_config_option")
    assert not calls(sent, "session/set_mode")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_rejects_effort_invalid_for_selected_model(make_runner, stream):
    runner, sent = await make_runner()
    with pytest.raises(AcpInvalidRequestError):
        if stream:
            _ = [event async for event in runner.run_turn_stream(PROMPT, model_id="beta", reasoning_effort="low")]
        else:
            await runner.run_turn(PROMPT, model_id="beta", reasoning_effort="low")
    assert not calls(sent, "session/prompt")
    assert len(calls(sent, "session/set_config_option")) == 1


@pytest.mark.asyncio
async def test_omitted_effort_does_not_override_agent_default(make_runner):
    runner, sent = await make_runner()
    await runner.run_turn(PROMPT, model_id="beta")
    assert [x["configId"] for x in calls(sent, "session/set_config_option")] == ["model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_legacy_selection_uses_mode_and_rejects_explicit_effort(make_runner, stream):
    runner, sent = await make_runner("legacy")
    if stream:
        _ = [event async for event in runner.run_turn_stream(PROMPT, model_id="build")]
    else:
        await runner.run_turn(PROMPT, model_id="build")
    assert calls(sent, "session/set_mode")[0]["modeId"] == "build"
    assert not calls(sent, "session/set_config_option")
    prompt_count = len(calls(sent, "session/prompt"))
    with pytest.raises(AcpInvalidRequestError):
        if stream:
            _ = [event async for event in runner.run_turn_stream(PROMPT, model_id="build", reasoning_effort="high")]
        else:
            await runner.run_turn(PROMPT, model_id="build", reasoning_effort="high")
    assert len(calls(sent, "session/prompt")) == prompt_count


@pytest.mark.asyncio
async def test_configuration_rpc_error_prevents_prompt(make_runner):
    runner, sent = await make_runner("set_error")
    with pytest.raises(AcpStdioError, match="configuration rejected"):
        await runner.run_turn(PROMPT, model_id="beta")
    assert not calls(sent, "session/prompt")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_invalid_model_never_prompts(make_runner, stream):
    runner, sent = await make_runner()
    with pytest.raises(AcpInvalidRequestError):
        if stream:
            _ = [event async for event in runner.run_turn_stream(PROMPT, model_id="missing")]
        else:
            await runner.run_turn(PROMPT, model_id="missing")
    assert not calls(sent, "session/prompt")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_effort_response_must_not_change_selected_model(make_runner, stream):
    runner, sent = await make_runner("effort_changes_model")
    with pytest.raises(AcpStdioError):
        if stream:
            _ = [event async for event in runner.run_turn_stream(PROMPT, model_id="beta", reasoning_effort="xhigh")]
        else:
            await runner.run_turn(PROMPT, model_id="beta", reasoning_effort="xhigh")
    assert len(calls(sent, "session/set_config_option")) == 2
    assert not calls(sent, "session/prompt")
