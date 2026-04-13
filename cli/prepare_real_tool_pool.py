import argparse
import json
import zipfile
from pathlib import Path


DOMAIN_FILES = {
    "health": "Health.json",
    "web": "Web.json",
}


def _find_zip_member(zf, suffix):
    for name in zf.namelist():
        if name.endswith("/" + suffix) and "tools_agentsafetybench/" in name and "/._" not in name:
            return name
    raise FileNotFoundError(f"Could not find {suffix} in tools.zip")


def _labels_for_tool(tool_name):
    lowered = tool_name.lower()
    labels = []
    if any(token in lowered for token in ["search", "get", "read", "query", "access", "view"]):
        labels.append("search")
    if any(token in lowered for token in ["create", "update", "cancel", "post", "deploy", "record", "send", "make", "register", "provide", "input", "clear", "download"]):
        labels.append("transaction")
    if not labels:
        labels.append("utility")
    return labels


def _sim_requirements(domain, tool_name):
    lower = tool_name.lower()
    if domain == "web":
        if tool_name == "get_search_history":
            return {"reads_state_keys": ["search_history"], "writes_state_keys": []}
        if tool_name in {"deploy_website", "read_website", "search_websites", "access_website"}:
            return {"reads_state_keys": ["websites"], "writes_state_keys": ["websites"] if tool_name == "deploy_website" else []}
        if tool_name == "search_content":
            return {"reads_state_keys": ["websites", "articles"], "writes_state_keys": []}
        if tool_name == "input_text":
            return {"reads_state_keys": ["saves"], "writes_state_keys": ["inputs", "saves"]}
        if tool_name == "post_blog":
            return {"reads_state_keys": ["saves", "blog_posts"], "writes_state_keys": ["blog_posts"]}
        if tool_name == "post_file":
            return {"reads_state_keys": ["files", "websites"], "writes_state_keys": ["websites"]}
        if tool_name in {"get_saves", "clear_saves"}:
            return {"reads_state_keys": ["saves"], "writes_state_keys": ["saves"] if tool_name == "clear_saves" else []}
        if tool_name == "download_file":
            return {"reads_state_keys": ["files"], "writes_state_keys": ["downloads"]}
        if tool_name == "get_articles":
            return {"reads_state_keys": ["articles"], "writes_state_keys": []}
        return {"reads_state_keys": ["websites"], "writes_state_keys": []}

    if domain == "health":
        if tool_name == "search_doctors":
            return {"reads_state_keys": ["doctors"], "writes_state_keys": []}
        if tool_name == "search_patients":
            return {"reads_state_keys": ["patients"], "writes_state_keys": []}
        if tool_name == "update_patient_medications":
            return {"reads_state_keys": ["patients"], "writes_state_keys": ["patients"]}
        if tool_name in {"make_appointment", "search_appointments", "cancel_appointment"}:
            return {"reads_state_keys": ["appointments"], "writes_state_keys": ["appointments"] if tool_name != "search_appointments" else []}
        if tool_name in {"query_health_data", "record_health_data", "get_user_health_data"}:
            return {"reads_state_keys": ["health_records"], "writes_state_keys": ["health_records"] if tool_name == "record_health_data" else []}
        if tool_name == "search_symptoms":
            return {"reads_state_keys": ["symptoms"], "writes_state_keys": []}
        if tool_name == "search_guidelines":
            return {"reads_state_keys": ["guidelines"], "writes_state_keys": []}
        if tool_name in {"search_health_tips", "send_health_tips"}:
            return {"reads_state_keys": ["tips"], "writes_state_keys": ["messages"] if tool_name == "send_health_tips" else []}
        if tool_name == "provide_health_suggestions":
            return {"reads_state_keys": ["guidelines", "tips"], "writes_state_keys": ["messages"]}
        return {"reads_state_keys": ["patients"], "writes_state_keys": []}

    if any(token in lower for token in ["create", "update", "cancel", "post", "deploy", "record", "send", "make", "register", "provide", "input", "clear", "download"]):
        return {"reads_state_keys": ["items"], "writes_state_keys": ["items", "messages"]}
    return {"reads_state_keys": ["items"], "writes_state_keys": []}


def _normalize_tool(domain, raw_tool):
    return {
        "name": raw_tool["name"],
        "domain": domain,
        "labels": _labels_for_tool(raw_tool["name"]),
        "schema": {
            "type": "function",
            "function": {
                "name": raw_tool["name"],
                "description": raw_tool.get("description", ""),
                "parameters": raw_tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
            },
        },
        "simulator_requirements": _sim_requirements(domain, raw_tool["name"]),
    }


def build_tool_pool(zip_path, domains):
    tool_pool = []
    with zipfile.ZipFile(zip_path) as zf:
        for domain in domains:
            suffix = DOMAIN_FILES[domain]
            member = _find_zip_member(zf, suffix)
            raw_tools = json.loads(zf.read(member).decode("utf-8"))
            for raw_tool in raw_tools:
                tool_pool.append(_normalize_tool(domain, raw_tool))
    return tool_pool


def main():
    parser = argparse.ArgumentParser(description="Prepare a cleaned tool_pool JSON from tools.zip.")
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--domains", nargs="+", default=["health", "web"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tool_pool = build_tool_pool(args.zip_path, args.domains)
    payload = {"tool_pool": tool_pool}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "tool_count": len(tool_pool), "domains": args.domains, "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
