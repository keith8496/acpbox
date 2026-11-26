"""Minimal newline-delimited ACP agent used by runner integration tests."""
import json
import os
import sys


def option(option_id, category, values, current):
    return {"id": option_id, "name": option_id, "category": category, "type": "select",
            "currentValue": current, "options": [{"value": x, "name": x} for x in values]}


def options(efforts, model="alpha", effort=None):
    return [option("model", "model", ["alpha", "beta"], model),
            option("thought_level", "thought_level", efforts, effort or efforts[0])]


scenario = os.environ.get("FAKE_ACP_SCENARIO", "modern")
model = "alpha"
effort = "low"
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    params = request.get("params", {})
    result = {}
    error = None
    if method == "session/new":
        result = {"sessionId": "s1", "modes": {"currentModeId": "plan", "availableModes": [
            {"id": "plan", "name": "Plan"}, {"id": "build", "name": "Build"}]}}
        if scenario != "legacy":
            result["configOptions"] = options(["low", "medium"])
            if scenario != "config_only":
                result["models"] = {"currentModelId": "alpha", "availableModels": [
                    {"modelId": "alpha", "name": "Alpha"}, {"modelId": "beta", "name": "Beta"}]}
            if scenario == "models_only":
                result.pop("configOptions")
            if scenario == "conflicting_models":
                result["models"]["availableModels"] = [{"modelId": "alpha-high", "name": "Alpha high"}]
    elif method == "session/set_config_option":
        if scenario == "set_error":
            error = {"code": -32602, "message": "configuration rejected"}
        else:
            if params.get("configId") == "model":
                model = params["value"]
                effort = "high" if model == "beta" else "low"
            else:
                effort = params["value"]
                if scenario == "effort_changes_model":
                    model = "alpha"
            result = {"configOptions": options(["high", "xhigh"] if model == "beta" or scenario == "effort_changes_model" else ["low", "medium"], model, effort)}
    elif method == "session/prompt":
        print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": "s1", "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "hello"}}}}), flush=True)
        result = {"stopReason": "end_turn"}
    response = {"jsonrpc": "2.0", "id": request.get("id")}
    response["error" if error else "result"] = error or result
    print(json.dumps(response), flush=True)
