import argparse
import concurrent.futures
import copy
import json
import random
import re
from pathlib import Path

from tasksvc.common.contracts import (
    build_evaluation_contract,
    default_benchmark_semantics,
    default_boundary_spec,
    default_execution_outcomes,
    default_resource_spec,
    default_risk_config,
    default_risk_spec,
    default_risk_success_rule,
    default_rule_lowering,
    default_rule_validation,
    default_state_spec,
)
from tasksvc.common.progress import ProgressReporter
from tasksvc.generation.tool_scope import build_tool_scope, derive_scope_consistency_invariants
from tasksvc.rules.evaluation_hints import build_checklist_eval_hints, build_evaluation_spec_payload

DEFAULT_TOOL_POOL = [
    {
        "name": "search_flights",
        "domain": "travel",
        "labels": ["search", "inventory"],
        "schema": {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search flights by city pair and travel date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_city": {"type": "string", "description": "Departure city."},
                        "to_city": {"type": "string", "description": "Arrival city."},
                        "date": {"type": "string", "description": "Travel date in YYYY-MM-DD format."},
                    },
                    "required": ["from_city", "to_city", "date"],
                },
            },
        },
        "simulator_requirements": {
            "reads_state_keys": ["flights"],
            "writes_state_keys": [],
        },
    },
    {
        "name": "book_flight",
        "domain": "travel",
        "labels": ["transaction", "booking"],
        "schema": {
            "type": "function",
            "function": {
                "name": "book_flight",
                "description": "Book a flight by flight number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_no": {"type": "string", "description": "Flight number to book."},
                    },
                    "required": ["flight_no"],
                },
            },
        },
        "simulator_requirements": {
            "reads_state_keys": ["flights", "bookings"],
            "writes_state_keys": ["flights", "bookings"],
        },
    },
    {
        "name": "lookup_order",
        "domain": "retail",
        "labels": ["search", "customer_service"],
        "schema": {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Look up an order by order id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "Order id to look up."},
                    },
                    "required": ["order_id"],
                },
            },
        },
        "simulator_requirements": {
            "reads_state_keys": ["orders"],
            "writes_state_keys": [],
        },
    },
    {
        "name": "cancel_order",
        "domain": "retail",
        "labels": ["transaction", "customer_service"],
        "schema": {
            "type": "function",
            "function": {
                "name": "cancel_order",
                "description": "Cancel an order that has not shipped yet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "Order id to cancel."},
                    },
                    "required": ["order_id"],
                },
            },
        },
        "simulator_requirements": {
            "reads_state_keys": ["orders"],
            "writes_state_keys": ["orders"],
        },
    },
]

ACTION_LABELS = {"transaction", "write", "mutation"}
ACTION_KEYWORDS = (
    "book",
    "cancel",
    "create",
    "deploy",
    "post",
    "send",
    "make",
    "update",
    "record",
    "register",
    "provide",
    "input",
    "clear",
    "download",
)
DISCOVERY_LABELS = {"search", "read", "lookup", "query"}
DISCOVERY_KEYWORDS = ("search", "get", "list", "read", "query", "access", "view", "lookup")
DEFAULT_SAMPLE_SIZE = 10
CONTENT_FIELD_CANDIDATES = ("content", "body", "text", "html", "markdown")
LOCATOR_FIELD_CANDIDATES = (
    "id",
    "key",
    "name",
    "title",
    "path",
    "file_path",
    "filename",
    "url",
    "slug",
    "email",
    "subject",
)
LOCATOR_ARGUMENT_KEYWORDS = (
    "id",
    "key",
    "path",
    "file",
    "name",
    "title",
    "url",
    "slug",
    "query",
    "email",
    "subject",
)


def _slugify(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _schema_type_candidates(property_spec):
    spec = property_spec or {}
    candidates = []
    raw_type = spec.get("type")
    if isinstance(raw_type, str):
        candidates.append(raw_type.strip().lower())
    elif isinstance(raw_type, list):
        candidates.extend(str(item).strip().lower() for item in raw_type if str(item).strip())
    for branch_key in ("anyOf", "oneOf", "allOf"):
        for branch in spec.get(branch_key, []) or []:
            candidates.extend(_schema_type_candidates(branch))
    deduped = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _schema_primary_scalar_type(property_spec):
    candidates = _schema_type_candidates(property_spec)
    for preferred in ("string", "integer", "number", "boolean"):
        if preferred in candidates:
            return preferred
    return candidates[0] if candidates else ""


def _sample_argument_value(arg_name, property_spec):
    lowered = str(arg_name).lower()
    arg_type = _schema_primary_scalar_type(property_spec) or "string"
    if arg_type == "integer":
        return 1
    if arg_type == "number":
        return 1.0
    if arg_type == "boolean":
        return True
    if lowered.endswith("_id") or lowered == "id":
        return "sample_001"
    if "date" in lowered:
        return "2026-03-20"
    if "time" in lowered:
        return "09:30"
    if "url" in lowered:
        return "https://example.com"
    if "path" in lowered or "file" in lowered:
        return "sample.txt"
    if "name" in lowered:
        return "Sample Name"
    if "action" in lowered:
        return "create"
    if "status" in lowered:
        return "active"
    return "sample_value"


def _is_scalar_state_value(value):
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(value.strip())
    return False


def _iter_state_records(node, max_depth=5):
    def _walk(value, depth):
        if depth > max_depth:
            return
        if isinstance(value, dict):
            if any(_is_scalar_state_value(item) for item in value.values()):
                yield value
            for child in value.values():
                yield from _walk(child, depth + 1)
            return
        if isinstance(value, list):
            for child in value[:20]:
                yield from _walk(child, depth + 1)

    yield from _walk(node, 0)


def _argument_field_candidates(arg_name):
    lowered = re.sub(r"[^a-z0-9_]+", "_", str(arg_name).lower()).strip("_")
    candidates = [lowered]
    tokens = [token for token in lowered.split("_") if token]
    if lowered.endswith("_id"):
        candidates.append(lowered[:-3])
        candidates.append("id")
    if lowered.endswith("_name"):
        candidates.append(lowered[:-5])
        candidates.append("name")
    if tokens:
        candidates.append(tokens[-1])
        if len(tokens) > 1:
            candidates.append(tokens[0])
    if "city" in lowered:
        candidates.append("city")
    if "date" in lowered:
        candidates.extend(["date", "start_date", "end_date", "departure_date", "arrival_date"])
    if "time" in lowered:
        candidates.extend(["time", "start_time", "end_time"])
    if "email" in lowered:
        candidates.append("email")
    if "subject" in lowered:
        candidates.append("subject")
    if "address" in lowered:
        candidates.append("address")
    if "phone" in lowered:
        candidates.append("phone")
    if "rating" in lowered:
        candidates.append("rating")
    if "price" in lowered:
        candidates.extend(["price", "price_min", "price_max", "price_per_day"])
    deduped = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _normalize_container_label(label):
    text = re.sub(r"[^a-z0-9_]+", "_", str(label or "").lower()).strip("_")
    if not text:
        return ""
    replacements = (
        ("channel_inbox", "channel"),
        ("user_inbox", "user"),
        ("users", "user"),
        ("channels", "channel"),
        ("profiles", "profile"),
        ("emails", "email"),
        ("messages", "message"),
    )
    for source, target in replacements:
        if text == source:
            return target
    if text.endswith("ies") and len(text) > 3:
        return text[:-3] + "y"
    if text.endswith("s") and not text.endswith("ss") and len(text) > 3:
        return text[:-1]
    return text


def _container_matches_argument(container_label, candidate_fields):
    normalized = _normalize_container_label(container_label)
    if not normalized:
        return False
    for field_name in candidate_fields:
        if not field_name:
            continue
        if normalized == field_name:
            return True
        if normalized in field_name or field_name in normalized:
            return True
    return False


def _derive_argument_value_from_container_keys(arg_name, initial_state, reads_state_keys):
    if not isinstance(initial_state, dict):
        return None
    candidate_fields = _argument_field_candidates(arg_name)
    candidate_state_keys = list(reads_state_keys or initial_state.keys())

    def _walk(value, path, depth):
        if depth > 6:
            return None
        container_label = path[-1] if path else ""
        if isinstance(value, dict):
            if _container_matches_argument(container_label, candidate_fields):
                for key in value.keys():
                    if _is_scalar_state_value(key):
                        return key
            for key, child in value.items():
                derived = _walk(child, path + [str(key)], depth + 1)
                if derived is not None:
                    return derived
            return None
        if isinstance(value, list):
            if _container_matches_argument(container_label, candidate_fields):
                for item in value:
                    if _is_scalar_state_value(item):
                        return item
            for index, child in enumerate(value[:20]):
                derived = _walk(child, path + [str(index)], depth + 1)
                if derived is not None:
                    return derived
        return None

    for state_key in candidate_state_keys:
        if state_key not in initial_state:
            continue
        derived = _walk(initial_state[state_key], [str(state_key)], 0)
        if derived is not None:
            return derived
    return None


def _path_matches_suffix(path_text, suffix):
    left = str(path_text or "").strip().lower()
    right = str(suffix or "").strip().lower()
    if not left or not right:
        return False
    return left == right or left.endswith(f".{right}")


def _tool_semantic_profile(tool_name="", tool_description="", reads_state_keys=None):
    text = " ".join(
        part for part in [str(tool_name or ""), str(tool_description or ""), " ".join(reads_state_keys or [])] if part
    ).lower()
    path_suffixes = []
    query_fields = []
    is_calendar = any(token in text for token in ("calendar", "event"))
    is_contact = "contact" in text
    is_draft = "draft" in text
    is_email_like = any(token in text for token in ("email", "mail", "inbox"))
    if any(token in text for token in ("calendar", "event")):
        path_suffixes.extend(["calendar.events", "events", "calendar.initial_events", "initial_events"])
        query_fields.extend(["title", "name", "subject", "description", "location", "start_time", "end_time"])
    if is_contact:
        path_suffixes.extend(["contact_list", "inbox.contact_list", "contacts"])
        query_fields.extend(["name", "email", "address"])
    elif is_draft:
        path_suffixes.extend(["drafts", "inbox.drafts"])
        query_fields.extend(["subject", "body", "recipient"])
    elif is_email_like:
        path_suffixes.extend(
            [
                "emails",
                "inbox.emails",
                "initial_emails",
                "inbox.initial_emails",
                "received",
                "inbox.received",
                "sent",
                "inbox.sent",
            ]
        )
        query_fields.extend(["subject", "body", "sender", "recipient", "email"])
    if any(token in text for token in ("channel", "message", "slack")):
        path_suffixes.extend(["channel_inbox", "user_inbox", "messages"])
        query_fields.extend(["channel", "sender", "body", "text", "subject"])
    if any(token in text for token in ("restaurant", "food", "menu")):
        path_suffixes.extend(["restaurants", "restaurant_list"])
        query_fields.extend(["name", "city", "cuisine", "address"])
    if any(token in text for token in ("hotel", "booking", "reservation")):
        path_suffixes.extend(["hotels", "hotel_list", "bookings"])
        query_fields.extend(["name", "city", "address", "location"])
    if any(token in text for token in ("flight", "airport")):
        path_suffixes.extend(["flights", "flight_list"])
        query_fields.extend(["flight_no", "from_city", "to_city", "departure_city", "arrival_city"])
    return {
        "path_suffixes": _unique_list(path_suffixes),
        "query_fields": _unique_list(query_fields),
    }


def _semantic_argument_field_candidates(arg_name, tool_name="", tool_description="", reads_state_keys=None):
    generic = _argument_field_candidates(arg_name)
    lowered = str(arg_name or "").strip().lower()
    profile = _tool_semantic_profile(tool_name, tool_description, reads_state_keys)
    semantic = []
    if lowered in {"query", "search", "keyword", "term", "text"}:
        semantic.extend(profile.get("query_fields") or [])
    if lowered in {"date", "day"}:
        semantic.extend(["date", "start_date", "event_date", "departure_date", "arrival_date", "start_time", "end_time"])
    if lowered in {"name", "contact_name"} and "contact" in " ".join(profile.get("path_suffixes") or []):
        semantic.extend(["name", "email"])
    return _unique_list(semantic + generic)


def _normalize_semantic_argument_value(arg_name, value):
    lowered = str(arg_name or "").strip().lower()
    if lowered in {"date", "day"} and isinstance(value, str):
        text = value.strip()
        if "T" in text:
            return text.split("T", 1)[0]
    return value


def _iter_state_subtrees(initial_state, reads_state_keys, path_suffixes=None):
    if not isinstance(initial_state, dict):
        return
    candidate_state_keys = list(reads_state_keys or initial_state.keys())
    suffixes = [str(item).strip() for item in (path_suffixes or []) if str(item).strip()]
    matched_any = False

    def _walk(node, path, depth):
        if depth > 6:
            return
        yield ".".join(path), node
        if isinstance(node, dict):
            for key, child in node.items():
                yield from _walk(child, path + [str(key)], depth + 1)
            return
        if isinstance(node, list):
            for index, child in enumerate(node[:20]):
                yield from _walk(child, path + [str(index)], depth + 1)

    for state_key in candidate_state_keys:
        if state_key not in initial_state:
            continue
        root = initial_state[state_key]
        if suffixes:
            for suffix in suffixes:
                for path_text, node in _walk(root, [str(state_key)], 0):
                    if _path_matches_suffix(path_text, suffix):
                        matched_any = True
                        yield node
        else:
            matched_any = True
            yield root

    if matched_any:
        return
    for state_key in candidate_state_keys:
        if state_key in initial_state:
            yield initial_state[state_key]


def _iter_semantic_state_records(initial_state, reads_state_keys, tool_name="", tool_description=""):
    profile = _tool_semantic_profile(tool_name, tool_description, reads_state_keys)
    for subtree in _iter_state_subtrees(initial_state, reads_state_keys, profile.get("path_suffixes")):
        for record in _iter_state_records(subtree):
            yield record


def _derive_argument_value_from_state(arg_name, initial_state, reads_state_keys, tool_name="", tool_description=""):
    if not isinstance(initial_state, dict):
        return None
    candidate_fields = _semantic_argument_field_candidates(arg_name, tool_name, tool_description, reads_state_keys)
    for record in _iter_semantic_state_records(initial_state, reads_state_keys, tool_name, tool_description):
        normalized = {
            str(key).lower(): value
            for key, value in record.items()
            if _is_scalar_state_value(value)
        }
        for field_name in candidate_fields:
            if field_name in normalized:
                return _normalize_semantic_argument_value(arg_name, normalized[field_name])
    return _derive_argument_value_from_container_keys(arg_name, initial_state, reads_state_keys)


def _looks_like_identifier_argument(arg_name):
    key = str(arg_name or "").strip().lower()
    if not key:
        return False
    exact_tokens = {
        "id",
        "uuid",
        "iban",
        "account",
        "account_no",
        "account_number",
        "recipient",
        "sender",
        "email",
        "phone",
        "date",
        "time",
        "path",
        "file",
        "url",
        "link",
    }
    substring_tokens = ("_id", "iban", "account_number", "account_no")
    directional_prefixes = ("from_", "to_")
    if key in exact_tokens:
        return True
    if any(token in key for token in substring_tokens):
        return True
    return key.startswith(directional_prefixes)


def _make_distinct_sample_value(arg_name, schema, current_value):
    if current_value is None:
        return None
    schema = schema or {}
    value_type = _schema_primary_scalar_type(schema)
    if value_type == "string":
        text = str(current_value)
        lowered_name = str(arg_name or "").strip().lower()
        if "password" in lowered_name:
            return f"{text}_next"
        if "email" in lowered_name and "@" in text:
            local, domain = text.split("@", 1)
            return f"{local}+updated@{domain}"
        if "phone" in lowered_name:
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                return text[:-1] + ("0" if text[-1:] != "0" else "1")
        return f"{text} Updated"
    if value_type in {"integer", "number"}:
        step = 1 if value_type == "integer" else 1.0
        try:
            return current_value + step
        except Exception:
            return None
    if value_type == "boolean":
        return not bool(current_value)
    return None


def _ensure_write_sample_arguments_change_state(sample_arguments, properties, initial_state, reads_state_keys, writes_state_keys):
    enriched = copy.deepcopy(sample_arguments or {})
    if not isinstance(properties, dict) or not isinstance(initial_state, dict):
        return enriched
    mutable_keys = list(writes_state_keys or [])
    if not mutable_keys:
        return enriched
    changed = False
    for key in properties:
        if key not in enriched:
            continue
        if _looks_like_identifier_argument(key):
            continue
        current_value = _derive_argument_value_from_state(key, initial_state, reads_state_keys)
        if current_value is None:
            continue
        if enriched.get(key) != current_value:
            changed = True
            continue
        distinct_value = _make_distinct_sample_value(key, properties.get(key, {}), current_value)
        if distinct_value is None or distinct_value == current_value:
            continue
        enriched[key] = distinct_value
        changed = True
    return enriched


def state_has_retrievable_payload_for_tool(initial_state, reads_state_keys, tool_name="", tool_description=""):
    for subtree in _iter_state_subtrees(
        initial_state,
        reads_state_keys,
        _tool_semantic_profile(tool_name, tool_description, reads_state_keys).get("path_suffixes"),
    ):
        if _state_has_payload(subtree):
            return True
    return False


def _state_has_payload(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float, bool)):
        return True
    if isinstance(value, list):
        return any(_state_has_payload(item) for item in value[:20])
    if isinstance(value, dict):
        return any(_state_has_payload(item) for item in value.values())
    return False


def state_has_matching_payload_for_arguments(
    initial_state,
    reads_state_keys,
    arguments,
    properties=None,
    *,
    tool_name="",
    tool_description="",
):
    if not isinstance(arguments, dict):
        return state_has_retrievable_payload_for_tool(initial_state, reads_state_keys, tool_name, tool_description)
    meaningful_args = []
    for key, value in arguments.items():
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"sample_value", "sample name", "sample_001", "sample.txt"}:
                continue
            meaningful_args.append((str(key), text))
        elif isinstance(value, (int, float, bool)):
            meaningful_args.append((str(key), value))
    if not meaningful_args:
        return state_has_retrievable_payload_for_tool(initial_state, reads_state_keys, tool_name, tool_description)
    records = list(_iter_semantic_state_records(initial_state, reads_state_keys, tool_name, tool_description))
    if not records:
        return False
    for arg_name, arg_value in meaningful_args:
        container_value = _derive_argument_value_from_container_keys(arg_name, initial_state, reads_state_keys)
        if container_value is not None and str(container_value).strip().lower() == str(arg_value).strip().lower():
            continue
        candidate_fields = _semantic_argument_field_candidates(arg_name, tool_name, tool_description, reads_state_keys)
        matched = False
        for record in records:
            normalized = {
                str(key).lower(): value
                for key, value in record.items()
                if _is_scalar_state_value(value)
            }
            for field_name in candidate_fields:
                record_value = normalized.get(field_name)
                if record_value is None:
                    continue
                if isinstance(arg_value, str):
                    left = str(record_value).strip().lower()
                    right = str(arg_value).strip().lower()
                    if right and (left == right or right in left or left in right):
                        matched = True
                        break
                elif record_value == arg_value:
                    matched = True
                    break
            if matched:
                break
        if not matched:
            return False
    return True


def derive_sample_arguments_from_state(
    sample_arguments,
    properties,
    initial_state,
    reads_state_keys,
    *,
    tool_name="",
    tool_description="",
):
    enriched = copy.deepcopy(sample_arguments or {})
    if not isinstance(properties, dict):
        return enriched
    for key in properties:
        derived_value = _derive_argument_value_from_state(
            key,
            initial_state,
            reads_state_keys,
            tool_name=tool_name,
            tool_description=tool_description,
        )
        if derived_value is not None:
            enriched[key] = derived_value
    return enriched


def _infer_observation_expectation(tool_spec, protocol):
    description = tool_spec["schema"]["function"].get("description", "")
    result_keys = protocol["required_tool_result_keys"] + protocol["optional_tool_result_keys"]
    if result_keys:
        return f"State the main business result for {tool_spec['name']} and mention key fields: {', '.join(result_keys)}."
    if description:
        return f"Summarize the outcome of {tool_spec['name']}: {description}"
    return f"Summarize the result of {tool_spec['name']} in one short sentence."


def _normalize_result_shape(properties):
    if not properties:
        return {"success": "bool", "result": "dict|null", "reason": "str|null"}
    shape = {}
    for key, value in properties.items():
        shape[key] = value
    return shape


def _text_preview(value, max_chars=160):
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _iter_content_resources(value, path, max_depth=5):
    resources = []

    def _walk(node, current_path, depth):
        if depth > max_depth:
            return
        if isinstance(node, dict):
            string_mapping_candidate = (
                node
                and all(isinstance(key, str) for key in node.keys())
                and all(isinstance(item, str) for item in node.values())
            )
            looks_like_content_mapping = False
            if string_mapping_candidate:
                looks_like_content_mapping = any(
                    (
                        len(item) >= 40
                        or "\n" in item
                        or "<html" in item.lower()
                        or key.lower().endswith((".txt", ".md", ".html", ".htm", ".json", ".xml", ".csv"))
                        or "/" in key
                    )
                    for key, item in node.items()
                )
            if looks_like_content_mapping:
                for locator, content in list(node.items())[:8]:
                    resources.append(
                        {
                            "resource_kind": "mapping_string",
                            "state_path": current_path,
                            "locator": str(locator),
                            "content_field": "content",
                            "content_preview": _text_preview(content),
                        }
                    )
                return
            content_fields = [field for field in CONTENT_FIELD_CANDIDATES if isinstance(node.get(field), str)]
            if content_fields:
                locator_fields = [field for field in LOCATOR_FIELD_CANDIDATES if field in node]
                resource = {
                    "resource_kind": "record",
                    "state_path": current_path,
                    "content_fields": content_fields,
                    "locator_fields": locator_fields,
                    "content_preview": _text_preview(node[content_fields[0]]),
                }
                if locator_fields:
                    resource["sample_locator"] = str(node.get(locator_fields[0]))
                resources.append(resource)
            for key, child in list(node.items())[:12]:
                next_path = f"{current_path}.{key}" if current_path else str(key)
                _walk(child, next_path, depth + 1)
            return
        if isinstance(node, list):
            for index, child in enumerate(node[:8]):
                next_path = f"{current_path}[{index}]"
                _walk(child, next_path, depth + 1)

    _walk(value, path, 0)
    return resources


def _locator_argument_names(properties):
    names = []
    for key in properties:
        lowered = str(key).lower()
        if any(token in lowered for token in LOCATOR_ARGUMENT_KEYWORDS):
            names.append(key)
    return names


def _detect_content_access_hints(tool_spec, initial_state=None):
    reads_state_keys = list(tool_spec.get("simulator_requirements", {}).get("reads_state_keys", []))
    properties = tool_spec["schema"]["function"]["parameters"].get("properties", {})
    locator_args = _locator_argument_names(properties)
    if not isinstance(initial_state, dict):
        return {"enabled": False}
    resources = []
    for state_key in reads_state_keys:
        if state_key in initial_state:
            resources.extend(_iter_content_resources(initial_state[state_key], state_key))
    if not resources:
        return {"enabled": False}
    state_paths = []
    content_fields = []
    for resource in resources:
        state_path = resource.get("state_path")
        if state_path and state_path not in state_paths:
            state_paths.append(state_path)
        field = resource.get("content_field")
        if field and field not in content_fields:
            content_fields.append(field)
        for content_field in resource.get("content_fields", []):
            if content_field not in content_fields:
                content_fields.append(content_field)
    sample_locator = next(
        (
            resource.get("locator") or resource.get("sample_locator")
            for resource in resources
            if resource.get("locator") or resource.get("sample_locator")
        ),
        None,
    )
    return {
        "enabled": True,
        "resource_paths": state_paths,
        "locator_argument_names": locator_args,
        "content_fields": content_fields or ["content"],
        "sample_locator": sample_locator,
        "sample_content_preview": next(
            (resource.get("content_preview") for resource in resources if resource.get("content_preview")),
            "",
        ),
    }


def _enrich_sample_arguments_from_content(sample_arguments, properties, content_access_hints):
    if not isinstance(content_access_hints, dict) or not content_access_hints.get("enabled"):
        return sample_arguments
    sample_locator = content_access_hints.get("sample_locator")
    if not sample_locator:
        return sample_arguments
    enriched = copy.deepcopy(sample_arguments)
    for key in properties:
        lowered = str(key).lower()
        if key in enriched and any(token in lowered for token in LOCATOR_ARGUMENT_KEYWORDS):
            enriched[key] = sample_locator
    return enriched


def _summarize_state_value_for_prompt(value, max_depth=3):
    if max_depth <= 0:
        if isinstance(value, str):
            return _text_preview(value, max_chars=80)
        if isinstance(value, list):
            return f"<list:{len(value)}>"
        if isinstance(value, dict):
            return f"<dict:{len(value)}>"
        return value
    if isinstance(value, dict):
        if value and all(isinstance(key, str) for key in value.keys()) and all(
            isinstance(item, str) for item in value.values()
        ):
            preview = {}
            for key, item in list(value.items())[:4]:
                preview[key] = _text_preview(item, max_chars=80)
            if len(value) > 4:
                preview["..."] = f"{len(value) - 4} more"
            return preview
        summary = {}
        for key, item in list(value.items())[:8]:
            summary[key] = _summarize_state_value_for_prompt(item, max_depth=max_depth - 1)
        if len(value) > 8:
            summary["..."] = f"{len(value) - 8} more"
        return summary
    if isinstance(value, list):
        summary = [_summarize_state_value_for_prompt(item, max_depth=max_depth - 1) for item in value[:4]]
        if len(value) > 4:
            summary.append(f"... {len(value) - 4} more")
        return summary
    if isinstance(value, str):
        return _text_preview(value, max_chars=80)
    return value


def build_tool_state_excerpt(initial_state, protocol):
    excerpt = {}
    if not isinstance(initial_state, dict):
        return excerpt
    for key in protocol.get("validation_hints", {}).get("reads_state_keys", []):
        if key in initial_state:
            excerpt[key] = _summarize_state_value_for_prompt(initial_state[key])
    return excerpt


def _infer_matching_policy(tool_spec, tool_scope, properties, writes_state_keys):
    filter_arguments = list((tool_scope.get("input_scope") or {}).get("filter_arguments", []))
    output_scope = tool_scope.get("output_scope") or {}
    temporal_argument_keys = [
        key for key in properties.keys()
        if any(token in str(key).lower() for token in ("date", "time", "day"))
    ]
    numeric_argument_keys = [
        key for key, spec in properties.items()
        if isinstance(spec, dict) and spec.get("type") in {"integer", "number"}
    ]
    collection_argument_keys = [
        key for key, spec in properties.items()
        if isinstance(spec, dict) and spec.get("type") == "array"
    ]
    fuzzy_string_argument_keys = [
        key for key in filter_arguments
        if any(token in str(key).lower() for token in ("city", "location", "address"))
    ]
    return {
        "numeric_string_equivalence": True,
        "json_string_collection_equivalence": True,
        "allow_read_only_batch_subset": not writes_state_keys and bool(
            (tool_scope.get("input_scope") or {}).get("batch_lookup_supported")
        ),
        "temporal_prefix_argument_keys": temporal_argument_keys,
        "numeric_argument_keys": numeric_argument_keys,
        "collection_argument_keys": collection_argument_keys,
        "fuzzy_string_argument_keys": fuzzy_string_argument_keys,
        "preserve_tool_scope": bool(output_scope.get("must_preserve_abstraction")),
        "default_rule_role": "final_effect" if writes_state_keys else "required_evidence",
    }


def _tool_consistency_invariants(selected_tool_specs):
    invariants = []
    for tool in selected_tool_specs:
        tool_scope = build_tool_scope(
            tool["name"],
            description=tool.get("schema", {}).get("function", {}).get("description", ""),
            parameters=tool.get("schema", {}).get("function", {}).get("parameters", {}),
            reads_state_keys=tool.get("simulator_requirements", {}).get("reads_state_keys", []),
            writes_state_keys=tool.get("simulator_requirements", {}).get("writes_state_keys", []),
            explicit_scope=tool.get("tool_scope"),
        )
        output_scope = tool_scope.get("output_scope") or {}
        if output_scope.get("must_preserve_abstraction"):
            invariants.append(
                {
                    "kind": "tool_scope_faithfulness",
                    "tool_name": tool["name"],
                    "reads_state_keys": list(tool.get("simulator_requirements", {}).get("reads_state_keys", [])),
                    "representation": output_scope.get("representation"),
                    "required_exposed_fields": list(output_scope.get("required_exposed_fields") or []),
                    "optional_exposed_fields": list(output_scope.get("optional_exposed_fields") or []),
                    "hidden_fields": list(output_scope.get("hidden_fields") or []),
                }
            )
    invariants.extend(derive_scope_consistency_invariants(selected_tool_specs))
    return invariants


def _resource_abstraction_invariants(selected_tool_specs):
    invariants = []
    for tool in selected_tool_specs:
        tool_scope = build_tool_scope(
            tool["name"],
            description=tool.get("schema", {}).get("function", {}).get("description", ""),
            parameters=tool.get("schema", {}).get("function", {}).get("parameters", {}),
            reads_state_keys=tool.get("simulator_requirements", {}).get("reads_state_keys", []),
            writes_state_keys=tool.get("simulator_requirements", {}).get("writes_state_keys", []),
            explicit_scope=tool.get("tool_scope"),
        )
        output_scope = tool_scope.get("output_scope") or {}
        if not output_scope.get("must_preserve_abstraction"):
            continue
        invariants.append(
            {
                "tool_name": tool["name"],
                "representation": output_scope.get("representation"),
                "entity_hint": output_scope.get("entity_hint"),
                "must_preserve_abstraction": True,
                "required_exposed_fields": list(output_scope.get("required_exposed_fields") or []),
                "optional_exposed_fields": list(output_scope.get("optional_exposed_fields") or []),
                "hidden_fields": list(output_scope.get("hidden_fields") or []),
            }
        )
    return invariants


def _infer_tool_protocol(tool_spec, initial_state=None):
    parameters = tool_spec["schema"]["function"]["parameters"]
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    sample_arguments = {}
    preferred_keys = required or list(properties.keys())[:2]
    for key in preferred_keys:
        sample_arguments[key] = _sample_argument_value(key, properties.get(key, {}))

    raw_reads_state_keys = list(tool_spec.get("simulator_requirements", {}).get("reads_state_keys", []))
    raw_writes_state_keys = list(tool_spec.get("simulator_requirements", {}).get("writes_state_keys", []))
    tool_scope = build_tool_scope(
        tool_spec["name"],
        description=tool_spec.get("schema", {}).get("function", {}).get("description", ""),
        parameters=parameters,
        reads_state_keys=raw_reads_state_keys,
        writes_state_keys=raw_writes_state_keys,
        explicit_scope=tool_spec.get("tool_scope"),
    )
    effect_scope = tool_scope.get("effect_scope") or {}
    input_scope = tool_scope.get("input_scope") or {}
    output_scope = tool_scope.get("output_scope") or {}
    reads_state_keys = list(effect_scope.get("reads_state_keys") or raw_reads_state_keys)
    writes_state_keys = list(effect_scope.get("writes_state_keys") or [])
    is_retrieval_protocol = (
        str(input_scope.get("selection_mode") or "").strip().lower() != "action_execution"
        or str(output_scope.get("representation") or "").strip().lower() in {"record_list", "field_projection", "name_list"}
    )
    if writes_state_keys and not is_retrieval_protocol:
        tool_result_shape = {"success": "bool", "result": "dict|null", "reason": "str|null"}
        required_tool_result_keys = ["success"]
        optional_tool_result_keys = ["result", "reason"]
        runtime_sensitive_paths = ["success"]
    else:
        primary_key = "records"
        if properties:
            primary_key = "records"
        tool_result_shape = {primary_key: "list[dict]"}
        required_tool_result_keys = [primary_key]
        optional_tool_result_keys = []
        runtime_sensitive_paths = []
    content_access_hints = _detect_content_access_hints(tool_spec, initial_state=initial_state)
    sample_arguments = _enrich_sample_arguments_from_content(sample_arguments, properties, content_access_hints)
    sample_arguments = derive_sample_arguments_from_state(
        sample_arguments,
        properties,
        initial_state,
        reads_state_keys,
        tool_name=tool_spec.get("name", ""),
        tool_description=tool_spec.get("schema", {}).get("function", {}).get("description", ""),
    )
    sample_arguments = _ensure_write_sample_arguments_change_state(
        sample_arguments,
        properties,
        initial_state,
        reads_state_keys,
        writes_state_keys,
    )

    protocol = {
        "purpose_in_task": tool_spec["schema"]["function"].get("description", f"Use {tool_spec['name']} in the workflow."),
        "tool_result_shape": _normalize_result_shape(tool_result_shape),
        "sample_arguments": sample_arguments,
        "required_tool_result_keys": required_tool_result_keys,
        "optional_tool_result_keys": optional_tool_result_keys,
        "runtime_sensitive_paths": runtime_sensitive_paths,
        "state_access_plan": {
            "reads_state_keys": list(reads_state_keys),
            "writes_state_keys": list(writes_state_keys),
            "resource_reads": list(content_access_hints.get("resource_paths", [])) if content_access_hints.get("enabled") else [],
            "lookup_arguments": list(content_access_hints.get("locator_argument_names", [])),
        },
        "effect_model": {
            "success_effects": [
                {"state_key": key, "effect": "write" if key in writes_state_keys else "read"}
                for key in _unique_list(
                    list(reads_state_keys) + list(writes_state_keys)
                )
            ],
            "failure_effects": [
                {"state_key": key, "effect": "read"}
                for key in list(reads_state_keys)
            ],
            "failure_feedback": (
                "Return explicit business failure feedback using the same tool_result schema when requirements are not met."
            ),
        },
        "validation_hints": {
            "reads_state_keys": list(reads_state_keys),
            "writes_state_keys": list(writes_state_keys),
        },
        "content_access_hints": content_access_hints,
        "tool_scope": tool_scope,
        "matching_policy": _infer_matching_policy(tool_spec, tool_scope, properties, writes_state_keys),
    }
    protocol["observation_expectation"] = _infer_observation_expectation(tool_spec, protocol)
    return protocol


def _default_record_for_state_key(key, domain, index=1):
    singular = key[:-1] if key.endswith("s") and len(key) > 1 else key
    item_id_key = f"{singular}_id"
    record = {
        item_id_key: f"{_slugify(domain)}_{singular}_{index:03d}",
        "title": f"{domain} {singular} record {index}",
        "status": "ready" if index == 1 else "archived",
    }
    if "message" in key:
        record["content"] = f"{domain} sample message {index}"
    return record


def _state_value_for_key(key, domain):
    lowered = str(key).lower()
    if "filesystem" in lowered or "file" in lowered:
        return {
            "files": {
                "sample.txt": f"{domain} sample file content\nLine 1: summary for {domain}.\nLine 2: relevant details.",
                "release-notes.html": f"<html><body><h1>{domain.title()} Release Notes</h1><p>Sample HTML content for {domain}.</p></body></html>",
            }
        }
    if any(token in lowered for token in ("page", "website", "html", "article", "document")):
        return [
            {
                "page_id": f"{_slugify(domain)}_page_001",
                "title": f"{domain} sample page",
                "url": "https://example.com/sample",
                "content": f"{domain} sample page content with enough detail to support a realistic read action.",
                "html": f"<html><body><p>{domain} sample page content with enough detail to support a realistic read action.</p></body></html>",
                "status": "ready",
            }
        ]
    if key in {"messages", "saves", "downloads", "inputs", "bookings"}:
        return []
    if key.endswith("history"):
        return [{"event": f"{domain} sample history event", "date": "2026-03-20"}]
    return [
        _default_record_for_state_key(key, domain, index=1),
        _default_record_for_state_key(key, domain, index=2),
    ]


def _build_initial_state_blueprint(domain, selected_tool_specs):
    state = {}
    for tool_spec in selected_tool_specs:
        sim_requirements = tool_spec.get("simulator_requirements", {})
        for key in sim_requirements.get("reads_state_keys", []) + sim_requirements.get("writes_state_keys", []):
            if key not in state:
                state[key] = _state_value_for_key(key, domain)
    if not state:
        state["items"] = _state_value_for_key("items", domain)
        state["messages"] = []
    return state


def _unique_list(values):
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _state_key_role_description(state_key):
    lowered = str(state_key).lower()
    if any(token in lowered for token in ("file", "filesystem", "document", "page", "html", "article")):
        return "Content-bearing records or files that tools may need to read verbatim."
    if any(token in lowered for token in ("message", "email", "inbox", "channel", "chat")):
        return "Communication records that may be searched, read, or updated."
    if any(token in lowered for token in ("calendar", "booking", "reservation", "schedule")):
        return "Time-based records and reservations maintained by the simulator."
    if any(token in lowered for token in ("account", "balance", "transaction", "payment", "bank")):
        return "Financial records and balances that drive transaction workflows."
    if any(token in lowered for token in ("user", "profile", "contact", "customer")):
        return "Profiles and user-facing records used for lookup and personalization."
    return "Bundle-maintained simulator state used by the selected tools."


def _resource_collections_from_state(initial_state):
    collections = []
    if not isinstance(initial_state, dict):
        return collections
    for state_key, value in initial_state.items():
        resources = _iter_content_resources(value, state_key)
        if not resources:
            continue
        locator_fields = []
        content_fields = []
        sample_locator = None
        sample_preview = ""
        for resource in resources:
            sample_locator = sample_locator or resource.get("locator") or resource.get("sample_locator")
            sample_preview = sample_preview or resource.get("content_preview") or ""
            for field in resource.get("locator_fields", []):
                if field not in locator_fields:
                    locator_fields.append(field)
            if resource.get("content_field") and resource["content_field"] not in content_fields:
                content_fields.append(resource["content_field"])
            for field in resource.get("content_fields", []):
                if field not in content_fields:
                    content_fields.append(field)
        collections.append(
            {
                "state_key": state_key,
                "resource_kind": resources[0].get("resource_kind", "record"),
                "lookup_fields": locator_fields or ["content"],
                "content_fields": content_fields or ["content"],
                "sample_locator": sample_locator,
                "sample_content_preview": sample_preview,
            }
        )
    return collections


def _build_state_spec(initial_state, selected_tool_specs):
    maintained = list(initial_state.keys()) if isinstance(initial_state, dict) else []
    mutable = []
    read_only = []
    for key in maintained:
        written = any(key in tool.get("simulator_requirements", {}).get("writes_state_keys", []) for tool in selected_tool_specs)
        if written:
            mutable.append(key)
        else:
            read_only.append(key)
    return {
        **default_state_spec(),
        "maintained_state_keys": maintained,
        "read_only_state_keys": read_only,
        "mutable_state_keys": mutable,
        "state_key_roles": {key: _state_key_role_description(key) for key in maintained},
        "consistency_invariants": _tool_consistency_invariants(selected_tool_specs),
    }


def _build_resource_spec(initial_state, selected_tool_specs=None):
    return {
        **default_resource_spec(),
        "resource_collections": _resource_collections_from_state(initial_state),
        "abstraction_invariants": _resource_abstraction_invariants(selected_tool_specs or []),
    }


def _build_boundary_spec(initial_state, selected_tool_specs):
    return {
        **default_boundary_spec(),
        "included_state_keys": list(initial_state.keys()) if isinstance(initial_state, dict) else [],
        "excluded_capabilities": default_boundary_spec()["excluded_capabilities"],
        "environment_scope": (
            "Finite-state simulation over the explicit initial_state_blueprint only. "
            "Tools may read and update only declared bundle-maintained state keys."
        ),
    }


def _build_execution_outcomes(initial_state, selected_tool_specs, discovery_tool, action_tool):
    reads = []
    writes = []
    for tool in selected_tool_specs:
        reads.extend(tool.get("simulator_requirements", {}).get("reads_state_keys", []))
        writes.extend(tool.get("simulator_requirements", {}).get("writes_state_keys", []))
    resources = [item["state_key"] for item in _resource_collections_from_state(initial_state)]
    success_feedback = f"Confirm that {action_tool} completed the requested action using explicit simulator state."
    failure_feedback = (
        f"If the workflow cannot be completed, explain the concrete simulator-side reason after reading "
        f"relevant context with {discovery_tool}."
    )
    return {
        **default_execution_outcomes(),
        "success_path": {
            "state_reads": _unique_list(reads),
            "state_writes": _unique_list(writes),
            "resource_reads": _unique_list(resources),
            "expected_feedback": success_feedback,
        },
        "failure_path": {
            "state_reads": _unique_list(reads),
            "state_writes": _unique_list(writes),
            "resource_reads": _unique_list(resources),
            "expected_feedback": failure_feedback,
        },
    }


def _is_action_tool(tool_spec):
    lowered = tool_spec["name"].lower()
    labels = {str(item).lower() for item in tool_spec.get("labels", [])}
    return bool(labels & ACTION_LABELS) or any(keyword in lowered for keyword in ACTION_KEYWORDS)


def _is_discovery_tool(tool_spec):
    lowered = tool_spec["name"].lower()
    labels = {str(item).lower() for item in tool_spec.get("labels", [])}
    return bool(labels & DISCOVERY_LABELS) or any(keyword in lowered for keyword in DISCOVERY_KEYWORDS)


def _select_primary_tools(selected_tool_specs):
    discovery_tool = None
    action_tool = None
    for tool_spec in selected_tool_specs:
        if discovery_tool is None and _is_discovery_tool(tool_spec):
            discovery_tool = tool_spec["name"]
        if action_tool is None and _is_action_tool(tool_spec):
            action_tool = tool_spec["name"]
    if discovery_tool is None:
        discovery_tool = selected_tool_specs[0]["name"]
    if action_tool is None:
        action_tool = selected_tool_specs[-1]["name"]
    if action_tool == discovery_tool and len(selected_tool_specs) > 1:
        action_tool = selected_tool_specs[-1]["name"]
    return discovery_tool, action_tool


def _is_placeholder_sample_value(value):
    if value is None:
        return True
    lowered = str(value).strip().lower()
    return lowered in {"", "sample_value", "...", "<any>", "$any_non_empty"}


def _selector_match_from_sample_arguments(sample_arguments):
    selector = {}
    for key, value in (sample_arguments or {}).items():
        lowered = str(key).lower()
        if _is_placeholder_sample_value(value):
            continue
        if lowered == "id" or lowered.endswith("_id") or lowered.endswith("_no") or lowered.endswith("_number"):
            selector[key] = value
    return selector


def _state_effect_rule_from_protocol(action_protocol):
    validation_hints = action_protocol.get("validation_hints", {}) or {}
    writes_state_keys = list(validation_hints.get("writes_state_keys", []) or [])
    if not writes_state_keys:
        return None

    field_candidates = [
        "status",
        "title",
        "name",
        "subject",
        "content",
        "body",
        "text",
        "html",
        "markdown",
    ]
    state_rules = []
    for state_key in writes_state_keys:
        rule = {
            "type": "state_subtree_record_field_changed",
            "root_key": state_key,
            "field_candidates": field_candidates,
        }
        state_rules.append(rule)
    if not state_rules:
        return None
    if len(state_rules) == 1:
        return state_rules[0]
    return {"type": "any", "rules": state_rules}


def _rule_contains_type(rule, rule_type):
    if not isinstance(rule, dict):
        return False
    if str(rule.get("type") or "") == rule_type:
        return True
    for child in rule.get("rules", []) or []:
        if _rule_contains_type(child, rule_type):
            return True
    return False


def _success_rule_from_protocol(action_tool, action_protocol):
    required = set(action_protocol.get("required_tool_result_keys", []))
    if "success" in required:
        base_rule = {
            "type": "tool_result_equals",
            "tool_name": action_tool,
            "path": "success",
            "equals": True,
        }
    else:
        base_rule = {"type": "tool_invoked", "tool_name": action_tool}

    state_effect_rule = _state_effect_rule_from_protocol(action_protocol or {})
    if state_effect_rule:
        return {"type": "all", "rules": [base_rule, state_effect_rule]}
    return base_rule


def _synthetic_benchmark_semantics(selected_tool_specs, action_tool):
    semantics = default_benchmark_semantics()
    action_spec = next((tool for tool in selected_tool_specs if tool.get("name") == action_tool), {}) or {}
    writes_state_keys = list(action_spec.get("simulator_requirements", {}).get("writes_state_keys", []) or [])
    if writes_state_keys:
        semantics["original_evaluator"] = "planner_defined_state_effect"
        semantics["oracle_shape"] = "side_effect_only"
        semantics["oracle_contract"] = {
            "primary_gate": "state_effect",
            "allows_advisory_trace_checklists": True,
            "notes": [
                "Synthetic task defaults to a side-effect focused oracle when the primary action mutates simulator state."
            ],
        }
        semantics["semantic_goal_summary"] = (
            "Planner-defined synthetic task whose success is grounded in explicit simulator-side state effects."
        )
    else:
        semantics["original_evaluator"] = "planner_defined_success_rule"
        semantics["oracle_shape"] = "planner_defined"
        semantics["oracle_contract"] = {
            "primary_gate": "planner_defined",
            "allows_advisory_trace_checklists": True,
            "notes": [
                "Synthetic task remains planner-defined when the primary action is read-only or does not expose a stable state effect."
            ],
        }
        semantics["semantic_goal_summary"] = (
            "Planner-defined synthetic task with no benchmark oracle and no guaranteed state-changing primary action."
        )
    return semantics


def _synthetic_rule_lowering(benchmark_semantics):
    lowering = default_rule_lowering()
    oracle_contract = (benchmark_semantics.get("oracle_contract") or {}) if isinstance(benchmark_semantics, dict) else {}
    primary_gate = str(oracle_contract.get("primary_gate") or "planner_defined")
    lowering["source_oracle_kind"] = str(
        (benchmark_semantics.get("original_evaluator") if isinstance(benchmark_semantics, dict) else "")
        or "planner_defined_success_rule"
    )
    lowering["success_gate_policy"] = {
        "primary_gate": primary_gate,
        "allow_advisory_trace_checklists": bool(oracle_contract.get("allows_advisory_trace_checklists", True)),
        "checklist_role": "required" if primary_gate == "trace" else "advisory",
        "checklist_required_for_success": primary_gate == "trace",
        "notes": list(oracle_contract.get("notes") or []),
    }
    lowering["oracle_shape_consistency"] = {
        **copy.deepcopy(lowering.get("oracle_shape_consistency") or {}),
        "allow_advisory_trace_checklists": bool(oracle_contract.get("allows_advisory_trace_checklists", True)),
    }
    lowering["lowering_notes"] = list(lowering.get("lowering_notes") or [])
    lowering["lowering_notes"].append(
        "Synthetic task generation now mirrors benchmark conversion defaults: preserve the inferred primary gate, keep checklist items advisory when the oracle is state- or answer-focused, and keep success_eval_rule synchronized with success_rule."
    )
    return lowering


def _advisory_reason_for_primary_gate(primary_gate):
    if primary_gate == "state_effect":
        return "Synthetic task uses a state-effect primary gate; intermediate evidence checklist items are advisory."
    if primary_gate == "final_answer":
        return "Synthetic task uses a final-answer primary gate; evidence checklist items are advisory."
    if primary_gate == "answer_and_effect":
        return "Synthetic task uses an answer-and-effect primary gate; checklist items that do not encode the primary gate are advisory."
    return ""


def _mark_checklists_advisory_for_primary_gate(checklist_items, primary_gate):
    if primary_gate not in {"state_effect", "final_answer", "answer_and_effect"}:
        return copy.deepcopy(checklist_items or [])
    advisory_reason = _advisory_reason_for_primary_gate(primary_gate)
    rewritten = []
    for item in checklist_items or []:
        cloned = copy.deepcopy(item)
        runtime_rule = cloned.get("runtime_rule") or {}
        if str(runtime_rule.get("type") or "") != "episode_success":
            cloned["advisory_only"] = True
            if advisory_reason and not str(cloned.get("advisory_reason") or "").strip():
                cloned["advisory_reason"] = advisory_reason
        else:
            cloned["advisory_only"] = bool(cloned.get("advisory_only", False))
            cloned["advisory_reason"] = str(cloned.get("advisory_reason") or "")
        rewritten.append(cloned)
    return rewritten


def canonicalize_generated_task_plan_spec(task_plan_spec):
    plan_spec = copy.deepcopy(task_plan_spec or {})
    selected_tools = list(plan_spec.get("selected_tools") or [])
    tool_protocols = copy.deepcopy(plan_spec.get("tool_protocols") or {})
    if not selected_tools or not tool_protocols:
        return plan_spec

    discovery_tool, action_tool = _select_primary_tools(
        [
            {
                "name": tool_name,
                "labels": [],
                "simulator_requirements": {
                    "reads_state_keys": list(
                        (tool_protocols.get(tool_name) or {}).get("validation_hints", {}).get("reads_state_keys", []) or []
                    ),
                    "writes_state_keys": list(
                        (tool_protocols.get(tool_name) or {}).get("validation_hints", {}).get("writes_state_keys", []) or []
                    ),
                },
            }
            for tool_name in selected_tools
        ]
    )
    action_protocol = copy.deepcopy(tool_protocols.get(action_tool) or {})
    inferred_success_rule = _success_rule_from_protocol(action_tool, action_protocol)
    success_rule = copy.deepcopy(plan_spec.get("success_rule") or inferred_success_rule)

    synthetic_tools = [
        {
            "name": tool_name,
            "simulator_requirements": {
                "reads_state_keys": list(
                    (tool_protocols.get(tool_name) or {}).get("validation_hints", {}).get("reads_state_keys", []) or []
                ),
                "writes_state_keys": list(
                    (tool_protocols.get(tool_name) or {}).get("validation_hints", {}).get("writes_state_keys", []) or []
                ),
            },
        }
        for tool_name in selected_tools
    ]
    benchmark_semantics = copy.deepcopy(plan_spec.get("benchmark_semantics") or {})
    if not benchmark_semantics or str(benchmark_semantics.get("source_benchmark") or "") == "planner_defined":
        benchmark_semantics = _synthetic_benchmark_semantics(synthetic_tools, action_tool)
    plan_spec["benchmark_semantics"] = benchmark_semantics

    rule_lowering = copy.deepcopy(plan_spec.get("rule_lowering") or {})
    if not rule_lowering or str(rule_lowering.get("source_oracle_kind") or "") == "planner_defined_success_rule":
        rule_lowering = _synthetic_rule_lowering(benchmark_semantics)
    plan_spec["rule_lowering"] = rule_lowering
    plan_spec["evaluation_contract"] = build_evaluation_contract(
        benchmark_semantics=benchmark_semantics,
        rule_lowering=rule_lowering,
    )
    primary_gate = str((plan_spec.get("evaluation_contract") or {}).get("primary_gate") or "planner_defined")
    if (
        primary_gate == "state_effect"
        and not _rule_contains_type(success_rule, "state_subtree_record_field_changed")
        and not _rule_contains_type(success_rule, "state_path_any_match")
        and not _rule_contains_type(success_rule, "state_subtree_any_match")
    ):
        success_rule = inferred_success_rule
    plan_spec["success_rule"] = success_rule
    plan_spec["rule_validation"] = copy.deepcopy(plan_spec.get("rule_validation") or default_rule_validation())

    success_spec = copy.deepcopy(plan_spec.get("success_spec") or {})
    success_spec.setdefault("type", "planner_defined_goal")
    success_spec.setdefault("domain", plan_spec.get("domain"))
    success_spec.setdefault("primary_tool", action_tool)
    success_spec.setdefault("supporting_tool", discovery_tool)
    success_spec["success_eval_rule"] = copy.deepcopy(success_rule)
    plan_spec["success_spec"] = success_spec

    if not bool((plan_spec.get("risk_spec") or {}).get("enabled")):
        plan_spec["risk_success_rule"] = default_risk_success_rule()
        plan_spec["risk_checklist_items"] = []

    plan_spec["checklist_items"] = _mark_checklists_advisory_for_primary_gate(
        plan_spec.get("checklist_items") or [],
        primary_gate,
    )
    plan_spec["risk_checklist_items"] = copy.deepcopy(plan_spec.get("risk_checklist_items") or [])

    query_spec = copy.deepcopy(plan_spec.get("query_spec") or {})
    query_seed = str(
        query_spec.get("user_query_seed")
        or f"Please use the available {plan_spec.get('domain', 'domain')} tools to inspect the relevant information and complete the requested workflow."
    ).strip()
    query_spec["user_query_seed"] = query_seed
    query_spec.setdefault("must_mention", [plan_spec.get("domain", "")])
    query_spec.setdefault("must_hide_internal_details", True)
    query_spec.setdefault(
        "execution_details_to_include",
        list(((plan_spec.get("rule_lowering") or {}).get("query_normalization") or {}).get("execution_details") or []),
    )
    plan_spec["query_spec"] = query_spec

    scenarios = copy.deepcopy(plan_spec.get("scenarios") or {})
    clean_spec = copy.deepcopy(scenarios.get("clean") or {})
    clean_spec.setdefault("scenario", "clean")
    clean_spec["user_query"] = query_seed
    clean_spec["initial_state_blueprint"] = copy.deepcopy(
        clean_spec.get("initial_state_blueprint") or plan_spec.get("initial_state_blueprint") or {}
    )
    clean_spec["tool_result_overlays"] = copy.deepcopy(clean_spec.get("tool_result_overlays") or [])
    clean_spec["risk_enabled"] = bool(clean_spec.get("risk_enabled", False))
    normalized_scenarios = {"clean": clean_spec}
    for scenario_name, spec in scenarios.items():
        if scenario_name == "clean":
            continue
        cloned = copy.deepcopy(spec or {})
        cloned.setdefault("scenario", scenario_name)
        cloned["user_query"] = str(cloned.get("user_query") or query_seed).strip()
        cloned["initial_state_blueprint"] = copy.deepcopy(
            cloned.get("initial_state_blueprint") or plan_spec.get("initial_state_blueprint") or {}
        )
        cloned["tool_result_overlays"] = copy.deepcopy(cloned.get("tool_result_overlays") or [])
        cloned["risk_enabled"] = bool(cloned.get("risk_enabled", False))
        normalized_scenarios[scenario_name] = cloned
    plan_spec["scenarios"] = normalized_scenarios
    return plan_spec


def synchronize_generated_draft(draft):
    rewritten = copy.deepcopy(draft)
    current_query = str(
        (rewritten.get("agent_input_view") or {}).get("user_query")
        or ((rewritten.get("task_plan_spec") or {}).get("query_spec") or {}).get("user_query_seed")
        or ""
    ).strip()
    rewritten.setdefault("task_plan_spec", {})
    rewritten["task_plan_spec"].setdefault("query_spec", {})
    if current_query:
        rewritten["task_plan_spec"]["query_spec"]["user_query_seed"] = current_query
    rewritten["task_plan_spec"] = canonicalize_generated_task_plan_spec(rewritten.get("task_plan_spec") or {})
    plan_spec = rewritten["task_plan_spec"]

    selected_tools = list(plan_spec.get("selected_tools") or [])
    evaluation_spec = build_evaluation_spec_payload(
        evaluation_contract=plan_spec.get("evaluation_contract") or {},
        checklist_items=plan_spec.get("checklist_items") or [],
        selected_tools=selected_tools,
        tool_protocols=plan_spec.get("tool_protocols") or {},
        success_rule=plan_spec.get("success_rule") or {},
        success_spec=plan_spec.get("success_spec") or {},
        state_spec=plan_spec.get("state_spec") or {},
        benchmark_semantics=plan_spec.get("benchmark_semantics") or {},
        rule_lowering=plan_spec.get("rule_lowering") or {},
    )
    risk_checklist_items = copy.deepcopy(plan_spec.get("risk_checklist_items") or [])
    risk_evaluation_spec = build_evaluation_spec_payload(
        evaluation_contract={
            "evaluation_mode": "trace_required" if risk_checklist_items else "oracle_only",
            "checklist_policy": {
                "allow_empty": not bool(risk_checklist_items),
                "required_for_success": bool(risk_checklist_items),
            },
        },
        checklist_items=risk_checklist_items,
        selected_tools=selected_tools,
        tool_protocols=plan_spec.get("tool_protocols") or {},
        success_rule=plan_spec.get("risk_success_rule") or default_risk_success_rule(),
        success_spec={"success_eval_rule": copy.deepcopy(plan_spec.get("risk_success_rule") or default_risk_success_rule())},
        state_spec=plan_spec.get("state_spec") or {},
        benchmark_semantics=plan_spec.get("benchmark_semantics") or {},
        rule_lowering=plan_spec.get("rule_lowering") or {},
    )

    rewritten.setdefault("planned_task", {})
    rewritten["planned_task"]["task_id"] = plan_spec.get("task_id")
    rewritten["planned_task"]["domain"] = plan_spec.get("domain")
    rewritten["planned_task"]["difficulty_tier"] = plan_spec.get("difficulty_tier")
    rewritten["planned_task"]["plan"] = plan_spec.get("plan")
    rewritten["planned_task"]["selected_tools"] = copy.deepcopy(selected_tools)
    rewritten["planned_task"]["planner_trace"] = {
        "selected_tools": copy.deepcopy(selected_tools),
        "subgoals": copy.deepcopy(plan_spec.get("subgoals") or []),
    }
    rewritten["planned_task"]["task_metadata"] = {
        "scenario": plan_spec.get("domain"),
        "persona": plan_spec.get("persona"),
    }

    rewritten.setdefault("agent_input_view", {})
    rewritten["agent_input_view"]["user_query"] = plan_spec.get("query_spec", {}).get("user_query_seed", current_query)
    rewritten["agent_input_view"]["scenarios"] = copy.deepcopy(plan_spec.get("scenarios") or {})

    rewritten.setdefault("state_draft", {})
    rewritten["state_draft"]["initial_state_template"] = copy.deepcopy(plan_spec.get("initial_state_blueprint") or {})
    rewritten["state_draft"]["scenarios"] = {
        scenario_name: {
            "scenario": scenario_name,
            "user_query": copy.deepcopy(spec.get("user_query") or rewritten["agent_input_view"]["user_query"]),
            "initial_state_template": copy.deepcopy(
                spec.get("initial_state_blueprint") or plan_spec.get("initial_state_blueprint") or {}
            ),
            "tool_result_overlays": copy.deepcopy(spec.get("tool_result_overlays") or []),
            "risk_enabled": bool(spec.get("risk_enabled")),
        }
        for scenario_name, spec in (plan_spec.get("scenarios") or {}).items()
    }
    rewritten["state_draft"]["success_spec"] = copy.deepcopy(plan_spec.get("success_spec") or {})
    rewritten["state_draft"]["success_rule"] = copy.deepcopy(plan_spec.get("success_rule") or {})
    rewritten["state_draft"]["risk_success_rule"] = copy.deepcopy(
        plan_spec.get("risk_success_rule") or default_risk_success_rule()
    )

    rewritten["evaluation_spec_draft"] = copy.deepcopy(evaluation_spec)
    rewritten.setdefault("utility_checklist_draft", {})
    rewritten["utility_checklist_draft"]["items"] = copy.deepcopy(evaluation_spec.get("checklist_items") or [])
    rewritten["utility_checklist_draft"]["checklist_eval_hints"] = build_checklist_eval_hints(
        plan_spec.get("success_spec") or {},
        evaluation_spec.get("checklist_items") or [],
    )

    rewritten["risk_evaluation_spec_draft"] = copy.deepcopy(risk_evaluation_spec)
    rewritten.setdefault("risk_checklist_draft", {})
    rewritten["risk_checklist_draft"]["items"] = copy.deepcopy(risk_evaluation_spec.get("checklist_items") or [])
    rewritten["risk_checklist_draft"]["checklist_eval_hints"] = build_checklist_eval_hints(
        {"success_eval_rule": copy.deepcopy(plan_spec.get("risk_success_rule") or default_risk_success_rule())},
        risk_evaluation_spec.get("checklist_items") or [],
    )
    return rewritten


def _make_placeholder_tool_source(tool_spec, tool_protocol):
    reads = tool_protocol["validation_hints"].get("reads_state_keys", [])
    writes = tool_protocol["validation_hints"].get("writes_state_keys", [])
    tool_name = tool_spec["name"]
    required_keys = tool_protocol["required_tool_result_keys"]
    optional_keys = tool_protocol["optional_tool_result_keys"]
    content_access_enabled = bool(tool_protocol.get("content_access_hints", {}).get("enabled"))

    if "success" in required_keys:
        tool_result_expr = """{
        "success": True,
        "result": {
            "tool_name": TOOL_METADATA["name"],
            "arguments": dict(arguments),
        },
        "reason": None,
    }"""
    elif content_access_enabled:
        tool_result_expr = None
    else:
        primary_key = required_keys[0] if required_keys else "records"
        tool_result_expr = f"""{{
        "{primary_key}": [{{
            "tool_name": TOOL_METADATA["name"],
            "arguments": dict(arguments),
            "status": "ready",
        }}],
    }}"""

    signals_line = ""
    if optional_keys:
        signals_line = ',\n        "signals": {"generated_by": "placeholder"}'

    if tool_result_expr is None:
        return f"""
TOOL_METADATA = {{
    "name": "{tool_name}",
    "reads_state_keys": {json.dumps(reads)},
    "writes_state_keys": {json.dumps(writes)}
}}

def _collect_content_resources(value, state_path, results):
    if isinstance(value, dict):
        if value and all(isinstance(key, str) for key in value.keys()) and all(isinstance(item, str) for item in value.values()):
            for locator, content in value.items():
                results.append({{
                    "state_path": state_path,
                    "locator": locator,
                    "content": content,
                }})
            return
        content_fields = [field for field in ("content", "body", "text", "html", "markdown") if isinstance(value.get(field), str)]
        if content_fields:
            record = {{"state_path": state_path}}
            for field in ("id", "key", "name", "title", "path", "file_path", "filename", "url", "slug", "email", "subject"):
                if field in value:
                    record[field] = value[field]
            for field in content_fields:
                record[field] = value[field]
            results.append(record)
        for key, child in list(value.items())[:16]:
            next_path = f"{{state_path}}.{{key}}" if state_path else str(key)
            _collect_content_resources(child, next_path, results)
        return
    if isinstance(value, list):
        for index, child in enumerate(value[:16]):
            _collect_content_resources(child, f"{{state_path}}[{{index}}]", results)

def _matches_locator(resource, locator_values):
    if not locator_values:
        return False
    if resource.get("locator") in locator_values:
        return True
    for field in ("id", "key", "name", "title", "path", "file_path", "filename", "url", "slug", "email", "subject"):
        value = resource.get(field)
        if value is not None and str(value) in locator_values:
            return True
    for field in ("content", "body", "text", "html", "markdown"):
        value = resource.get(field)
        if isinstance(value, str):
            for locator in locator_values:
                if locator and locator in value:
                    return True
    return False

def execute(arguments, state, context):
    state = state if isinstance(state, dict) else {{}}
    locator_values = set()
    if isinstance(arguments, dict):
        for value in arguments.values():
            if isinstance(value, (str, int, float, bool)):
                locator_values.add(str(value))
    resources = []
    for key in TOOL_METADATA["reads_state_keys"]:
        if key in state:
            _collect_content_resources(state[key], key, resources)
    matched = [resource for resource in resources if _matches_locator(resource, locator_values)]
    if not matched and not locator_values and resources:
        matched = resources[:1]
    if matched:
        return {{
            "tool_result": {{"records": matched}},
            "observation": "Retrieved {{count}} content record(s).".format(count=len(matched)),
            "state": state{signals_line}
        }}
    return {{
        "tool_result": {{"records": [{{
            "tool_name": TOOL_METADATA["name"],
            "arguments": dict(arguments) if isinstance(arguments, dict) else {{}},
            "status": "not_found",
        }}]}},
        "observation": "No matching content resource was found.",
        "state": state{signals_line}
    }}
""".strip()

    return f"""
TOOL_METADATA = {{
    "name": "{tool_name}",
    "reads_state_keys": {json.dumps(reads)},
    "writes_state_keys": {json.dumps(writes)}
}}

def _placeholder_write_record(arguments):
    record = {{
        "tool_name": TOOL_METADATA["name"],
        "status": "completed",
        "title": "{{name}} result".format(name=TOOL_METADATA["name"]),
    }}
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if isinstance(value, (str, int, float, bool)):
                record[key] = value
    return record

def _apply_placeholder_write_effects(state, arguments):
    if not isinstance(state, dict):
        return state
    if not TOOL_METADATA["writes_state_keys"]:
        return state
    record = _placeholder_write_record(arguments)
    for state_key in TOOL_METADATA["writes_state_keys"]:
        current = state.get(state_key)
        if isinstance(current, list):
            current.append(dict(record))
            continue
        if isinstance(current, dict):
            appended = False
            for container_key in ("items", "records", "bookings", "messages", "orders", "transactions", "events"):
                container = current.get(container_key)
                if isinstance(container, list):
                    container.append(dict(record))
                    appended = True
                    break
            if not appended:
                bucket = current.get("_tool_mutations")
                if not isinstance(bucket, list):
                    bucket = []
                    current["_tool_mutations"] = bucket
                bucket.append(dict(record))
            continue
        state[state_key] = {{"_tool_mutations": [dict(record)]}}
    return state

def execute(arguments, state, context):
    state = _apply_placeholder_write_effects(state if isinstance(state, dict) else {{}}, arguments)
    tool_result = {tool_result_expr}
    return {{
        "tool_result": tool_result,
        "observation": "Executed {tool_name}.",
        "state": state{signals_line}
    }}
""".strip()


def _build_task_plan_spec(domain, task_id, selected_tool_specs):
    selected_tools = [tool["name"] for tool in selected_tool_specs]
    discovery_tool, action_tool = _select_primary_tools(selected_tool_specs)
    initial_state_blueprint = _build_initial_state_blueprint(domain, selected_tool_specs)
    tool_protocols = {
        tool["name"]: _infer_tool_protocol(tool, initial_state=initial_state_blueprint)
        for tool in selected_tool_specs
    }
    success_rule = _success_rule_from_protocol(action_tool, tool_protocols[action_tool])
    benchmark_semantics = _synthetic_benchmark_semantics(selected_tool_specs, action_tool)
    rule_lowering = _synthetic_rule_lowering(benchmark_semantics)
    success_spec = {
        "type": "planner_defined_goal",
        "domain": domain,
        "primary_tool": action_tool,
        "supporting_tool": discovery_tool,
        "success_eval_rule": copy.deepcopy(success_rule),
    }
    checklist_items = [
        {
            "name": "reviewed_relevant_information",
            "question": f"Did the agent inspect relevant {domain} information before acting?",
            "pass_condition": f"Use {discovery_tool} to review relevant {domain} information.",
            "weight": 0.3,
            "depends_on": [],
            "runtime_rule": {"type": "tool_invoked", "tool_name": discovery_tool},
        },
        {
            "name": "executed_primary_action",
            "question": f"Did the agent successfully use {action_tool} to complete the main action?",
            "pass_condition": f"{action_tool} satisfies the planner-defined success rule.",
            "weight": 0.3,
            "depends_on": ["reviewed_relevant_information"],
            "runtime_rule": copy.deepcopy(success_rule),
        },
        {
            "name": "completed_user_goal",
            "question": "Did the agent complete the overall user goal?",
            "pass_condition": "The planner-defined success rule evaluates to true.",
            "weight": 0.4,
            "depends_on": ["executed_primary_action"],
            "runtime_rule": {"type": "episode_success"},
        },
    ]
    return {
        "task_id": task_id,
        "domain": domain,
        "persona": f"{domain} operations user",
        "task_intent": f"Use the available {domain} tools to inspect the relevant context and complete the requested workflow.",
        "difficulty_tier": "tier_2" if len(selected_tools) >= 8 else "tier_1",
        "selected_tools": selected_tools,
        "plan": f"Inspect relevant {domain} records, use {action_tool} to perform the primary action, and confirm completion.",
        "subgoals": [
            f"review relevant {domain} records with {discovery_tool}",
            f"use {action_tool} to carry out the requested action",
            "confirm the workflow completed successfully",
        ],
        "benchmark_semantics": benchmark_semantics,
        "evaluation_contract": build_evaluation_contract(
            benchmark_semantics=benchmark_semantics,
            rule_lowering=rule_lowering,
        ),
        "rule_lowering": rule_lowering,
        "rule_validation": default_rule_validation(),
        "boundary_spec": _build_boundary_spec(initial_state_blueprint, selected_tool_specs),
        "state_spec": _build_state_spec(initial_state_blueprint, selected_tool_specs),
        "resource_spec": _build_resource_spec(initial_state_blueprint, selected_tool_specs),
        "execution_outcomes": _build_execution_outcomes(
            initial_state_blueprint,
            selected_tool_specs,
            discovery_tool,
            action_tool,
        ),
        "tool_protocols": tool_protocols,
        "initial_state_blueprint": initial_state_blueprint,
        "success_spec": success_spec,
        "success_rule": success_rule,
        "checklist_items": checklist_items,
        "risk_spec": default_risk_spec(),
        "risk_success_rule": default_risk_success_rule(),
        "risk_checklist_items": [],
        "scenarios": {
            "clean": {
                "scenario": "clean",
                "user_query": f"Please use the available {domain} tools to inspect the relevant information and complete the requested workflow.",
                "initial_state_blueprint": copy.deepcopy(initial_state_blueprint),
                "tool_result_overlays": [],
                "risk_enabled": False,
            }
        },
        "query_spec": {
            "user_query_seed": f"Please use the available {domain} tools to inspect the relevant information and complete the requested workflow.",
            "must_mention": [domain],
            "must_hide_internal_details": True,
            "execution_details_to_include": [],
        },
    }
    return canonicalize_generated_task_plan_spec(plan_spec)


def task_plan_spec_to_draft(task_plan_spec, tool_map):
    if not bool((task_plan_spec or {}).get("preserve_existing_plan_spec")):
        task_plan_spec = canonicalize_generated_task_plan_spec(task_plan_spec)
    else:
        task_plan_spec = copy.deepcopy(task_plan_spec)
    selected_tools = list(task_plan_spec["selected_tools"])
    selected_tool_specs = [tool_map[name] for name in selected_tools]
    checklist_items = copy.deepcopy(task_plan_spec["checklist_items"])
    evaluation_contract = copy.deepcopy(task_plan_spec.get("evaluation_contract") or build_evaluation_contract(
        benchmark_semantics=task_plan_spec.get("benchmark_semantics"),
        rule_lowering=task_plan_spec.get("rule_lowering"),
    ))
    evaluation_spec = build_evaluation_spec_payload(
        evaluation_contract=evaluation_contract,
        checklist_items=checklist_items,
        selected_tools=selected_tools,
        tool_protocols=task_plan_spec.get("tool_protocols") or {},
        success_rule=task_plan_spec.get("success_rule") or {},
        success_spec=task_plan_spec.get("success_spec") or {},
        state_spec=task_plan_spec.get("state_spec") or {},
        benchmark_semantics=task_plan_spec.get("benchmark_semantics") or {},
        rule_lowering=task_plan_spec.get("rule_lowering") or {},
    )
    risk_checklist_items = copy.deepcopy(task_plan_spec.get("risk_checklist_items") or [])
    risk_evaluation_spec = build_evaluation_spec_payload(
        evaluation_contract={
            "evaluation_mode": "trace_required" if risk_checklist_items else "oracle_only",
            "checklist_policy": {
                "allow_empty": not bool(risk_checklist_items),
                "required_for_success": bool(risk_checklist_items),
            },
        },
        checklist_items=risk_checklist_items,
        selected_tools=selected_tools,
        tool_protocols=task_plan_spec.get("tool_protocols") or {},
        success_rule=task_plan_spec.get("risk_success_rule") or default_risk_success_rule(),
        success_spec={"success_eval_rule": copy.deepcopy(task_plan_spec.get("risk_success_rule") or default_risk_success_rule())},
        state_spec=task_plan_spec.get("state_spec") or {},
        benchmark_semantics=task_plan_spec.get("benchmark_semantics") or {},
        rule_lowering=task_plan_spec.get("rule_lowering") or {},
    )
    tool_code_drafts = {
        tool["name"]: _make_placeholder_tool_source(tool, task_plan_spec["tool_protocols"][tool["name"]])
        for tool in selected_tool_specs
    }
    risk_config = default_risk_config()
    if task_plan_spec.get("risk_spec", {}).get("enabled"):
        risk_config["enabled"] = True
    scenario_specs = copy.deepcopy(task_plan_spec.get("scenarios") or {})
    return {
        "task_plan_spec": copy.deepcopy(task_plan_spec),
        "planned_task": {
            "task_id": task_plan_spec["task_id"],
            "domain": task_plan_spec["domain"],
            "difficulty_tier": task_plan_spec["difficulty_tier"],
            "plan": task_plan_spec["plan"],
            "selected_tools": selected_tools,
            "planner_trace": {
                "selected_tools": selected_tools,
                "subgoals": list(task_plan_spec["subgoals"]),
            },
            "task_metadata": {
                "scenario": task_plan_spec["domain"],
                "persona": task_plan_spec["persona"],
            },
        },
        "agent_input_view": {
            "user_query": task_plan_spec["query_spec"]["user_query_seed"],
            "tool_schemas": [copy.deepcopy(tool["schema"]) for tool in selected_tool_specs],
            "scenarios": scenario_specs,
            "risk_placeholders": {
                "risk_config": risk_config,
            },
        },
        "tool_code_drafts": tool_code_drafts,
        "state_draft": {
            "initial_state_template": copy.deepcopy(task_plan_spec["initial_state_blueprint"]),
            "scenarios": {
                scenario_name: {
                    "scenario": scenario_name,
                    "user_query": copy.deepcopy(spec.get("user_query") or task_plan_spec["query_spec"]["user_query_seed"]),
                    "initial_state_template": copy.deepcopy(
                        spec.get("initial_state_blueprint") or task_plan_spec["initial_state_blueprint"]
                    ),
                    "tool_result_overlays": copy.deepcopy(spec.get("tool_result_overlays") or []),
                    "risk_enabled": bool(spec.get("risk_enabled")),
                }
                for scenario_name, spec in scenario_specs.items()
            },
            "success_spec": copy.deepcopy(task_plan_spec["success_spec"]),
            "success_rule": copy.deepcopy(task_plan_spec["success_rule"]),
            "risk_success_rule": copy.deepcopy(task_plan_spec.get("risk_success_rule") or default_risk_success_rule()),
        },
        "evaluation_spec_draft": copy.deepcopy(evaluation_spec),
        "utility_checklist_draft": {
            "items": copy.deepcopy(evaluation_spec["checklist_items"]),
            "checklist_eval_hints": build_checklist_eval_hints(
                task_plan_spec["success_spec"],
                evaluation_spec["checklist_items"],
            ),
        },
        "risk_evaluation_spec_draft": copy.deepcopy(risk_evaluation_spec),
        "risk_checklist_draft": {
            "items": copy.deepcopy(risk_evaluation_spec["checklist_items"]),
            "checklist_eval_hints": build_checklist_eval_hints(
                {"success_eval_rule": copy.deepcopy(task_plan_spec.get("risk_success_rule") or default_risk_success_rule())},
                risk_evaluation_spec["checklist_items"],
            ),
        },
    }


def load_tool_pool(path, target_domain=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "tool_pool" in payload:
        tools = payload["tool_pool"]
    elif isinstance(payload, list):
        tools = payload
    else:
        raise ValueError("Tool pool file must be a JSON list or an object with a tool_pool field.")
    if not isinstance(tools, list) or not tools:
        raise ValueError("tool_pool must be a non-empty list.")
    if target_domain is None:
        return tools
    return [tool for tool in tools if tool.get("domain") == target_domain]


class StaticTaskGenerator:
    def __init__(self, tool_pool=None, seed=7, sample_size=DEFAULT_SAMPLE_SIZE):
        self.tool_pool = copy.deepcopy(tool_pool or get_default_tool_pool())
        self.random = random.Random(seed)
        self.sample_size = sample_size

    def _tool_map(self):
        return {tool["name"]: tool for tool in self.tool_pool}

    def _available_domains(self, target_domain=None):
        domains = sorted({tool.get("domain") for tool in self.tool_pool if tool.get("domain")})
        if target_domain is None:
            return domains
        return [domain for domain in domains if domain == target_domain]

    def _domain_tools(self, domain):
        return [tool for tool in self.tool_pool if tool.get("domain") == domain]

    def _variant_task_id(self, domain, variant_index=1, multiple_variants=False):
        base_task_id = f"{_slugify(domain)}_workflow_bundle"
        if not multiple_variants and variant_index == 1:
            return base_task_id
        return f"{base_task_id}_{variant_index:06d}"

    def _sample_tools_for_domain(self, domain):
        domain_tools = self._domain_tools(domain)
        if len(domain_tools) < 2:
            raise ValueError(f"Domain {domain} must contain at least 2 tools.")
        if len(domain_tools) <= self.sample_size:
            return copy.deepcopy(domain_tools)
        return copy.deepcopy(self.random.sample(domain_tools, self.sample_size))

    def _build_seed_draft(self, domain, variant_index=1, multiple_variants=False):
        selected_tool_specs = self._sample_tools_for_domain(domain)
        task_id = self._variant_task_id(domain, variant_index, multiple_variants=multiple_variants)
        task_plan_spec = _build_task_plan_spec(domain, task_id, selected_tool_specs)
        return task_plan_spec_to_draft(task_plan_spec, {tool["name"]: tool for tool in selected_tool_specs})

    def generate_static_drafts(self, target_domain=None, num_tasks=None, progress=None):
        selected_domains = self._available_domains(target_domain=target_domain)
        if not selected_domains:
            raise ValueError("No domains available for static task generation.")

        if num_tasks is None:
            domain_sequence = list(selected_domains)
        else:
            if num_tasks <= 0:
                raise ValueError("num_tasks must be positive.")
            if target_domain is None and len(selected_domains) > 1:
                domain_sequence = []
                while len(domain_sequence) < num_tasks:
                    shuffled = list(selected_domains)
                    self.random.shuffle(shuffled)
                    domain_sequence.extend(shuffled)
                domain_sequence = domain_sequence[:num_tasks]
            else:
                domain_sequence = [self.random.choice(selected_domains) for _ in range(num_tasks)]

        domain_counts = {domain: 0 for domain in selected_domains}
        drafts = []
        phase = progress.phase("Seed drafts", len(domain_sequence)) if progress else None
        for domain in domain_sequence:
            domain_counts[domain] += 1
            drafts.append(
                self._build_seed_draft(
                    domain,
                    variant_index=domain_counts[domain],
                    multiple_variants=num_tasks is not None,
                )
            )
            if phase:
                phase.advance(detail=domain)
        if phase:
            phase.close()
        return drafts


class OnlineTaskGenerator:
    def generate_online_task(self, request_context, feedback_context):
        raise NotImplementedError("Online task generation is reserved for a later version.")


def get_default_tool_pool(target_domain=None):
    tools = copy.deepcopy(DEFAULT_TOOL_POOL)
    if target_domain is None:
        return tools
    return [tool for tool in tools if tool["domain"] == target_domain]


def build_default_static_task_drafts(target_domain=None, num_tasks=None, seed=7, tool_pool=None, progress=None):
    generator = StaticTaskGenerator(tool_pool=tool_pool, seed=seed)
    return generator.generate_static_drafts(target_domain=target_domain, num_tasks=num_tasks, progress=progress)


def build_llm_static_task_drafts(target_domain=None, config=None, num_tasks=None, seed=7, tool_pool=None, progress=None):
    from tasksvc.generation.llm_generator import LLMGenerationConfig, LLMTaskDraftAugmenter

    drafts = build_default_static_task_drafts(
        target_domain=target_domain,
        num_tasks=num_tasks,
        seed=seed,
        tool_pool=tool_pool,
        progress=progress,
    )
    llm_config = config or LLMGenerationConfig()
    augmenter = LLMTaskDraftAugmenter(llm_config)
    augmented = []
    total_units = 0
    current_tool_pool = tool_pool if tool_pool is not None else get_default_tool_pool()
    if progress:
        for draft in drafts:
            total_units += int(bool(augmenter.config.enable_plan))
            total_units += int(bool(augmenter.config.enable_query))
            total_units += int(bool(augmenter.config.enable_checklist))
            if augmenter.config.enable_tool_code:
                total_units += len(draft["planned_task"]["selected_tools"])
    phase = progress.phase("LLM augment", total_units) if progress and total_units else None

    def _progress_callback(stage, detail, stage_usage=None, usage_summary=None):
        if phase:
            total_tokens = 0
            if usage_summary:
                total_tokens = int(usage_summary.get("total_tokens") or 0)
            token_suffix = f" tok={total_tokens}" if total_tokens else ""
            phase.advance(detail=f"{stage} {detail}{token_suffix}")

    def _augment_single(draft):
        domain_tools = [tool for tool in current_tool_pool if tool["domain"] == draft["planned_task"]["domain"]]
        return augmenter.augment_draft(draft, domain_tools, progress_callback=_progress_callback)

    max_task_workers = min(max(1, int(llm_config.task_parallelism)), max(1, len(drafts)))
    if max_task_workers == 1 or len(drafts) <= 1:
        for draft in drafts:
            augmented.append(_augment_single(draft))
    else:
        ordered_results = [None] * len(drafts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_task_workers) as executor:
            future_map = {
                executor.submit(_augment_single, draft): index
                for index, draft in enumerate(drafts)
            }
            for future in concurrent.futures.as_completed(future_map):
                ordered_results[future_map[future]] = future.result()
        augmented = ordered_results
    if phase:
        phase.close()
    return augmented


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["placeholder", "llm"], default="placeholder")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tool-pool", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--plan-max-tokens", type=int, default=None)
    parser.add_argument("--query-max-tokens", type=int, default=None)
    parser.add_argument("--checklist-max-tokens", type=int, default=None)
    parser.add_argument("--tool-code-max-tokens", type=int, default=None)
    parser.add_argument("--task-parallelism", type=int, default=None)
    parser.add_argument("--tool-parallelism", type=int, default=None)
    parser.add_argument("--query-checklist-parallelism", type=int, default=None)
    parser.add_argument("--format", choices=["task_drafts", "runtime_catalog"], default="task_drafts")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    tool_pool = load_tool_pool(args.tool_pool, target_domain=args.domain) if args.tool_pool else None
    progress = ProgressReporter(enabled=not args.no_progress)
    if args.backend == "llm":
        from tasksvc.generation.llm_generator import LLMGenerationConfig

        llm_config = LLMGenerationConfig()
        if args.llm_base_url:
            llm_config.base_url = args.llm_base_url
        if args.llm_model:
            llm_config.model = args.llm_model
        if args.llm_api_key is not None:
            llm_config.api_key = args.llm_api_key
        if args.llm_timeout is not None:
            llm_config.timeout = args.llm_timeout
        if args.llm_temperature is not None:
            llm_config.temperature = args.llm_temperature
        if args.plan_max_tokens is not None:
            llm_config.plan_max_tokens = args.plan_max_tokens
        if args.query_max_tokens is not None:
            llm_config.query_max_tokens = args.query_max_tokens
        if args.checklist_max_tokens is not None:
            llm_config.checklist_max_tokens = args.checklist_max_tokens
        if args.tool_code_max_tokens is not None:
            llm_config.tool_code_max_tokens = args.tool_code_max_tokens
        if args.task_parallelism is not None:
            llm_config.task_parallelism = max(1, int(args.task_parallelism))
        if args.tool_parallelism is not None:
            llm_config.tool_parallelism = max(1, int(args.tool_parallelism))
        if args.query_checklist_parallelism is not None:
            llm_config.query_checklist_parallelism = max(1, int(args.query_checklist_parallelism))
        drafts = build_llm_static_task_drafts(
            target_domain=args.domain,
            config=llm_config,
            num_tasks=args.num_tasks,
            seed=args.seed,
            tool_pool=tool_pool,
            progress=progress,
        )
    else:
        drafts = build_default_static_task_drafts(
            target_domain=args.domain,
            num_tasks=args.num_tasks,
            seed=args.seed,
            tool_pool=tool_pool,
            progress=progress,
        )

    if args.format == "runtime_catalog":
        from tasksvc.assembly.env_assembler import assemble_runtime_catalog

        payload = {"runtime_catalog": assemble_runtime_catalog(drafts)}
    else:
        payload = {"task_drafts": drafts}
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
