import copy
import re


_COMMON_ENTITY_FIELDS = [
    "id",
    "name",
    "title",
    "recipient",
    "recipients",
    "sender",
    "city",
    "address",
    "rating",
    "reviews",
    "review_count",
    "price",
    "price_min",
    "price_max",
    "amount",
    "currency",
    "cuisine",
    "contact_information",
    "email",
    "phone",
    "website",
    "participants",
    "subject",
    "body",
    "content",
    "text",
    "html",
    "date",
    "start_time",
    "end_time",
    "location",
    "fuel_options",
    "car_types",
]

_RAW_CONTENT_FIELDS = ["content", "body", "text", "html", "locator", "path", "file_path", "name", "title"]
_PRICE_FIELDS = ["name", "price", "price_min", "price_max", "amount", "currency"]
_ADDRESS_FIELDS = ["name", "address", "location"]
_RATING_REVIEW_FIELDS = ["name", "rating", "reviews", "review_count"]
_CONTACT_FIELDS = ["name", "contact_information", "email", "phone", "website"]
_EMAIL_MESSAGE_FIELDS = ["sender", "recipient", "recipients", "subject", "body"]
_CHANNEL_MESSAGE_FIELDS = ["sender", "subject", "body", "content", "text", "date"]


def _unique_list(values):
    ordered = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def _normalize_field_label(label):
    text = str(label or "").strip().lower()
    if not text:
        return ""
    replacements = {
        "e-mail": "email",
        "email address": "email",
        "file id": "id",
        "message body": "body",
        "email body": "body",
        "recipient address": "recipient",
        "recipient list": "recipient",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\b(a|an|the|its|their|each|all|corresponding|matching|returned|available)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text.endswith("ies") and len(text) > 3:
        text = text[:-3] + "y"
    elif text.endswith("s") and not text.endswith("ss") and len(text) > 3:
        text = text[:-1]
    return text


def _fields_from_description(description):
    text = str(description or "")
    if not text:
        return []
    patterns = [
        r"Each [^.]*? has (?P<fields>[^.]+)\.",
        r"Each [^.]*? includes (?P<fields>[^.]+)\.",
        r"Each [^.]*? contains (?P<fields>[^.]+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        fields_blob = match.group("fields")
        fields_blob = re.sub(r"\band\b", ",", fields_blob, flags=re.IGNORECASE)
        fields = []
        for raw_part in fields_blob.split(","):
            normalized = _normalize_field_label(raw_part)
            if normalized:
                fields.append(normalized)
        deduped = _unique_list(fields)
        if deduped:
            return deduped
    return []


def _email_scope_fields(tool_name, description, properties):
    lowered = f"{tool_name} {description}".lower()
    prop_names = {str(name or "").lower() for name in (properties or {}).keys()}
    if "email" not in lowered and "inbox" not in lowered:
        return []
    if not (
        "search" in lowered
        or "sent email" in lowered
        or "received email" in lowered
        or "draft email" in lowered
        or "unread email" in lowered
        or "emails in the inbox" in lowered
    ):
        return []
    if "search" in lowered and "email" in lowered:
        return list(_EMAIL_MESSAGE_FIELDS)
    fields = []
    if "sender" in lowered or "sender" in prop_names or "received" in lowered or "unread" in lowered:
        fields.append("sender")
    if "recipient" in lowered or "recipient" in prop_names or "recipients" in prop_names or "sent" in lowered or "draft" in lowered:
        fields.extend(["recipient", "recipients"])
    if "subject" in lowered:
        fields.append("subject")
    if "body" in lowered or "content" in lowered:
        fields.append("body")
    return _unique_list(fields)


def _message_scope_fields(tool_name, description, properties):
    lowered = f"{tool_name} {description}".lower()
    prop_names = {str(name or "").lower() for name in (properties or {}).keys()}
    if not any(token in lowered for token in ("message", "channel", "thread", "slack", "chat", "inbox")):
        return []
    if not any(token in lowered for token in ("read", "search", "query", "fetch", "list")):
        return []
    fields = []
    if "sender" in lowered or "sender" in prop_names:
        fields.append("sender")
    if "subject" in lowered or "subject" in prop_names:
        fields.append("subject")
    if "body" in lowered or "content" in lowered or "text" in lowered:
        fields.extend(["body", "content", "text"])
    if "date" in lowered or "time" in lowered or "day" in lowered:
        fields.append("date")
    deduped = _unique_list(fields)
    if deduped:
        return deduped
    if "message" in lowered:
        return list(_CHANNEL_MESSAGE_FIELDS)
    return []


def _is_action_tool(tool_name, description, writes_state_keys):
    lowered = f"{tool_name} {description}".lower()
    action_tokens = ["create", "update", "cancel", "delete", "send", "share", "append", "write", "reserve"]
    retrieval_prefixes = ("get_", "list_", "search_", "find_", "read_", "fetch_", "query_")
    if any(str(tool_name or "").lower().startswith(prefix) for prefix in retrieval_prefixes):
        if not any(token in lowered for token in action_tokens):
            return False
    if writes_state_keys:
        return True
    return any(token in lowered for token in action_tokens)


def _entity_hint_from_name(tool_name):
    lowered = str(tool_name or "").lower()
    patterns = [
        (r"get_all_(?P<entity>.+?)_in_", "entity"),
        (r"get_rating_reviews_for_(?P<entity>.+)", "entity"),
        (r"get_(?P<entity>.+?)_prices?", "entity"),
        (r"get_(?P<entity>.+?)_address", "entity"),
        (r"get_contact_information_for_(?P<entity>.+)", "entity"),
    ]
    for pattern, group_name in patterns:
        match = re.match(pattern, lowered)
        if match:
            entity = match.group(group_name).strip("_")
            return entity.rstrip("s")
    keyword_hints = [
        ("channel", "channel"),
        ("message", "message"),
        ("email", "email"),
        ("inbox", "email"),
        ("file", "file"),
        ("directory", "file"),
        ("calendar", "calendar_event"),
        ("event", "calendar_event"),
        ("contact", "contact"),
        ("transaction", "transaction"),
        ("payment", "payment"),
        ("profile", "profile"),
    ]
    for token, hint in keyword_hints:
        if token in lowered:
            return hint
    return ""


def _lookup_entity_hint(tool_name, properties):
    lowered_name = str(tool_name or "").lower()
    property_names = [str(name or "").lower() for name in (properties or {}).keys()]
    mappings = [
        ("channel", "channel"),
        ("hotel", "hotel"),
        ("restaurant", "restaurant"),
        ("car", "car_rental"),
        ("flight", "flight"),
        ("email", "email"),
        ("message", "message"),
        ("contact", "contact"),
        ("file", "file"),
        ("event", "calendar_event"),
        ("calendar", "calendar_event"),
        ("transaction", "transaction"),
        ("payment", "payment"),
        ("profile", "profile"),
        ("user", "profile"),
    ]
    for prop_name in property_names:
        for token, hint in mappings:
            if token in prop_name:
                return hint
    for token, hint in mappings:
        if token in lowered_name:
            return hint
    return ""


def _heuristic_output_scope(tool_name, description, properties, writes_state_keys):
    lowered = f"{tool_name} {description}".lower()
    entity_hint = _entity_hint_from_name(tool_name)
    enumerated_fields = _fields_from_description(description)
    email_fields = _email_scope_fields(tool_name, description, properties)
    message_fields = _message_scope_fields(tool_name, description, properties)
    if _is_action_tool(tool_name, description, writes_state_keys):
        return {
            "representation": "operation_status",
            "entity_hint": entity_hint,
            "exposed_fields": [],
            "hidden_fields": [],
            "must_preserve_abstraction": False,
        }
    if any(token in lowered for token in ["read file", "read page", "read html", "document", "message body", "full raw content"]):
        return {
            "representation": "raw_content",
            "entity_hint": entity_hint,
            "exposed_fields": list(_RAW_CONTENT_FIELDS),
            "hidden_fields": [],
            "must_preserve_abstraction": True,
        }
    if enumerated_fields:
        hidden_fields = [field for field in _COMMON_ENTITY_FIELDS if field not in set(enumerated_fields)]
        return {
            "representation": "field_projection",
            "entity_hint": entity_hint,
            "exposed_fields": list(enumerated_fields),
            "required_exposed_fields": list(enumerated_fields),
            "optional_exposed_fields": [],
            "hidden_fields": hidden_fields,
            "must_preserve_abstraction": True,
            "record_shape": "object",
        }
    if email_fields:
        hidden_fields = [field for field in _COMMON_ENTITY_FIELDS if field not in set(email_fields)]
        return {
            "representation": "field_projection",
            "entity_hint": entity_hint or "email",
            "exposed_fields": list(email_fields),
            "required_exposed_fields": list(email_fields),
            "optional_exposed_fields": [],
            "hidden_fields": hidden_fields,
            "must_preserve_abstraction": True,
            "record_shape": "object",
        }
    if message_fields:
        hidden_fields = [field for field in _COMMON_ENTITY_FIELDS if field not in set(message_fields)]
        return {
            "representation": "field_projection",
            "entity_hint": entity_hint or "message",
            "exposed_fields": list(message_fields),
            "required_exposed_fields": list(message_fields),
            "optional_exposed_fields": [],
            "hidden_fields": hidden_fields,
            "must_preserve_abstraction": True,
            "record_shape": "object",
        }
    if re.match(r"^(get_all_|list_)", str(tool_name or "").lower()):
        hidden_fields = [field for field in _COMMON_ENTITY_FIELDS if field not in {"id", "name", "title"}]
        return {
            "representation": "name_list",
            "entity_hint": entity_hint,
            "exposed_fields": ["name"],
            "required_exposed_fields": ["name"],
            "optional_exposed_fields": ["id", "title"],
            "hidden_fields": hidden_fields,
            "must_preserve_abstraction": True,
            "record_shape": "scalar_or_named_record",
        }
    if "price" in lowered:
        hidden_fields = [field for field in _COMMON_ENTITY_FIELDS if field not in set(_PRICE_FIELDS)]
        return {
            "representation": "field_projection",
            "entity_hint": entity_hint,
            "exposed_fields": list(_PRICE_FIELDS),
            "required_exposed_fields": ["name"],
            "optional_exposed_fields": [field for field in _PRICE_FIELDS if field != "name"],
            "hidden_fields": hidden_fields,
            "must_preserve_abstraction": True,
            "record_shape": "object",
        }
    if "address" in lowered:
        hidden_fields = [field for field in _COMMON_ENTITY_FIELDS if field not in set(_ADDRESS_FIELDS)]
        return {
            "representation": "field_projection",
            "entity_hint": entity_hint,
            "exposed_fields": list(_ADDRESS_FIELDS),
            "required_exposed_fields": ["name", "address"],
            "optional_exposed_fields": ["location"],
            "hidden_fields": hidden_fields,
            "must_preserve_abstraction": True,
            "record_shape": "object",
        }
    if "rating" in lowered or "review" in lowered:
        hidden_fields = [field for field in _COMMON_ENTITY_FIELDS if field not in set(_RATING_REVIEW_FIELDS)]
        return {
            "representation": "field_projection",
            "entity_hint": entity_hint,
            "exposed_fields": list(_RATING_REVIEW_FIELDS),
            "required_exposed_fields": ["name", "rating"],
            "optional_exposed_fields": ["reviews", "review_count"],
            "hidden_fields": hidden_fields,
            "must_preserve_abstraction": True,
            "record_shape": "object",
        }
    if "contact" in lowered or "email" in lowered or "phone" in lowered or "website" in lowered:
        hidden_fields = [field for field in _COMMON_ENTITY_FIELDS if field not in set(_CONTACT_FIELDS)]
        return {
            "representation": "field_projection",
            "entity_hint": entity_hint,
            "exposed_fields": list(_CONTACT_FIELDS),
            "required_exposed_fields": ["name"],
            "optional_exposed_fields": [field for field in _CONTACT_FIELDS if field != "name"],
            "hidden_fields": hidden_fields,
            "must_preserve_abstraction": True,
            "record_shape": "object",
        }
    if any(token in lowered for token in ["read_", "search_", "query_", "fetch_"]) or properties:
        return {
            "representation": "record_list",
            "entity_hint": entity_hint,
            "exposed_fields": [],
            "required_exposed_fields": [],
            "optional_exposed_fields": [],
            "hidden_fields": [],
            "must_preserve_abstraction": False,
            "record_shape": "object",
        }
    return {
        "representation": "record_list",
        "entity_hint": entity_hint,
        "exposed_fields": [],
        "required_exposed_fields": [],
        "optional_exposed_fields": [],
        "hidden_fields": [],
        "must_preserve_abstraction": False,
        "record_shape": "object",
    }


def _merge_scope(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_scope(merged[key], value)
        elif isinstance(value, list):
            merged[key] = list(value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _should_apply_explicit_scope(explicit_scope):
    if not explicit_scope:
        return False
    source = str((explicit_scope or {}).get("scope_source") or "").strip().lower()
    return source not in {"", "heuristic_inference", "heuristic_refreshable"}


def build_tool_scope(
    tool_name,
    description="",
    parameters=None,
    reads_state_keys=None,
    writes_state_keys=None,
    explicit_scope=None,
):
    properties = ((parameters or {}).get("properties") or {}) if isinstance(parameters, dict) else {}
    required = list((parameters or {}).get("required") or []) if isinstance(parameters, dict) else []
    reads_state_keys = list(reads_state_keys or [])
    writes_state_keys = list(writes_state_keys or [])
    is_action = _is_action_tool(tool_name, description, writes_state_keys)
    effective_writes_state_keys = list(writes_state_keys if is_action else [])
    base = {
        "scope_source": "heuristic_inference",
        "input_scope": {
            "filter_arguments": _unique_list(required or list(properties.keys())[:2]),
            "selection_mode": "action_execution" if is_action else "argument_lookup",
            "lookup_entity_hint": _lookup_entity_hint(tool_name, properties),
            "batch_lookup_supported": any(
                isinstance((properties.get(key) or {}), dict) and (properties.get(key) or {}).get("type") == "array"
                for key in properties
            ),
        },
        "output_scope": _heuristic_output_scope(tool_name, description, properties, effective_writes_state_keys),
        "effect_scope": {
            "kind": (
                "read_write"
                if reads_state_keys and effective_writes_state_keys
                else "write_only"
                if effective_writes_state_keys
                else "read_only"
            ),
            "reads_state_keys": list(reads_state_keys),
            "writes_state_keys": list(effective_writes_state_keys),
            "side_effect_free": not bool(effective_writes_state_keys),
        },
        "faithfulness_notes": [
            "Preserve the original benchmark-visible abstraction level for this tool.",
        ] if not is_action else [],
    }
    if _should_apply_explicit_scope(explicit_scope):
        base = _merge_scope(base, explicit_scope)
        if explicit_scope.get("scope_source"):
            base["scope_source"] = explicit_scope["scope_source"]
        else:
            base["scope_source"] = "explicit_override"
    return base


def derive_scope_consistency_invariants(tool_specs):
    scopes = []
    for tool_spec in list(tool_specs or []):
        if not isinstance(tool_spec, dict) or not tool_spec.get("name"):
            continue
        schema = tool_spec.get("schema", {}).get("function", {})
        tool_scope = build_tool_scope(
            tool_spec["name"],
            description=schema.get("description", ""),
            parameters=schema.get("parameters", {}),
            reads_state_keys=tool_spec.get("simulator_requirements", {}).get("reads_state_keys", []),
            writes_state_keys=tool_spec.get("simulator_requirements", {}).get("writes_state_keys", []),
            explicit_scope=tool_spec.get("tool_scope"),
        )
        scopes.append(
            {
                "tool_name": tool_spec["name"],
                "scope": tool_scope,
            }
        )
    grouped = {}
    for item in scopes:
        scope = item["scope"]
        output_scope = scope.get("output_scope") or {}
        input_scope = scope.get("input_scope") or {}
        entity_hint = output_scope.get("entity_hint") or input_scope.get("lookup_entity_hint")
        if not entity_hint:
            continue
        bucket = grouped.setdefault(
            entity_hint,
            {
                "producers": [],
                "consumers": [],
                "read_state_keys": [],
                "lookup_arguments": [],
                "batch_lookup_supported_by": [],
            },
        )
        effect_scope = scope.get("effect_scope") or {}
        reads_state_keys = list(effect_scope.get("reads_state_keys") or [])
        filter_arguments = list(input_scope.get("filter_arguments") or [])
        for key in reads_state_keys:
            if key not in bucket["read_state_keys"]:
                bucket["read_state_keys"].append(key)
        for key in filter_arguments:
            if key not in bucket["lookup_arguments"]:
                bucket["lookup_arguments"].append(key)
        if input_scope.get("batch_lookup_supported") and item["tool_name"] not in bucket["batch_lookup_supported_by"]:
            bucket["batch_lookup_supported_by"].append(item["tool_name"])
        if output_scope.get("representation") in {"name_list", "field_projection", "record_list"}:
            bucket["producers"].append(item["tool_name"])
        if effect_scope.get("kind") in {"read_only", "read_write", "write_only"}:
            bucket["consumers"].append(item["tool_name"])
    invariants = []
    for entity_hint, bucket in grouped.items():
        producers = _unique_list(bucket.get("producers"))
        consumers = _unique_list(bucket.get("consumers"))
        if not producers or len(consumers) < 2:
            continue
        invariants.append(
            {
                "kind": "shared_lookup_visibility",
                "entity_hint": entity_hint,
                "producer_tools": producers,
                "consumer_tools": consumers,
                "read_state_keys": _unique_list(bucket.get("read_state_keys")),
                "lookup_arguments": _unique_list(bucket.get("lookup_arguments")),
                "batch_lookup_supported_by": _unique_list(bucket.get("batch_lookup_supported_by")),
                "notes": [
                    "Tools operating on the same entity set should agree on the visible entity universe and not surface "
                    "identifiers that peer tools cannot subsequently read or act upon.",
                ],
            }
        )
    return invariants
