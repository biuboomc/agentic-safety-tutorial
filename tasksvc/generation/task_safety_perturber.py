import copy
from dataclasses import asdict, dataclass, field


@dataclass
class TaskSafetyPerturbationConfig:
    enabled: bool = False
    mode: str = "reserved_not_implemented"
    unsafe_injection_probability: float = 0.25
    prompt_injection_slots: list[str] = field(default_factory=list)
    tool_injection_slots: list[str] = field(default_factory=list)


class TaskSafetyPerturber:
    def __init__(self, config=None):
        self.config = config or TaskSafetyPerturbationConfig()

    def apply_to_draft(self, task_draft):
        updated = copy.deepcopy(task_draft)
        risk_placeholders = updated.setdefault("agent_input_view", {}).setdefault("risk_placeholders", {})
        risk_placeholders["risk_config"] = {
            "enabled": self.config.enabled,
            "unsafe_injection_probability": self.config.unsafe_injection_probability,
            "prompt_injection_slots": list(self.config.prompt_injection_slots),
            "tool_injection_slots": list(self.config.tool_injection_slots),
        }
        updated["safety_perturbation_draft"] = {
            "reserved": True,
            "mode": self.config.mode,
            "status": "pass_through",
            "config": asdict(self.config),
            "notes": [
                "This module is intentionally reserved between task generation and runtime bundle assembly.",
                "No prompt or tool-response perturbation is applied in v1.",
            ],
        }
        return updated

    def apply_to_drafts(self, task_drafts):
        return [self.apply_to_draft(task_draft) for task_draft in task_drafts]
