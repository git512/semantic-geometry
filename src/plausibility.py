from __future__ import annotations

import json
import urllib.request

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .geometry import Concept


ROLE_SCORES = {
    "flow": {"gradient", "path", "signal"},
    "gradient": {"flow", "tension", "contrast"},
    "resistance": {"noise", "boundary", "constraint"},
    "power": {"transformation", "productivity"},
    "path": {"flow", "context", "channel"},
    "reference": {"context", "boundary"},
    "storage": {"memory", "capital"},
    "instability": {"noise", "conflict", "disturbance"},
    "context": {"reference", "path"},
    "boundary": {"reference", "resistance"},
    "transformation": {"power", "resolution"},
    "signal": {"flow", "meaning"},
    "noise": {"resistance", "instability"},
    "memory": {"storage", "reference"},
    "resolution": {"transformation", "power"},
    "tension": {"gradient", "instability"},
}


def cosine(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return float(cosine_similarity(vector_a.reshape(1, -1), vector_b.reshape(1, -1))[0, 0])


def role_match_score(role_a: str, role_b: str) -> float:
    role_a = role_a or "unknown"
    role_b = role_b or "unknown"
    if role_a == role_b and role_a != "unknown":
        return 1.0
    if role_b in ROLE_SCORES.get(role_a, set()) or role_a in ROLE_SCORES.get(role_b, set()):
        return 0.65
    if role_a == "unknown" or role_b == "unknown":
        return 0.5
    return 0.2


def rescale_cosine(value: float) -> float:
    return float(max(0.0, min(1.0, (value + 1.0) / 2.0)))


def score_mapping_plausibility(
    concept_a: Concept,
    concept_b: Concept,
    label_embedding_a: np.ndarray,
    label_embedding_b: np.ndarray,
    description_embedding_a: np.ndarray | None = None,
    description_embedding_b: np.ndarray | None = None,
    llm_judge_score: float | None = None,
) -> dict:
    label_similarity = rescale_cosine(cosine(label_embedding_a, label_embedding_b))
    if description_embedding_a is not None and description_embedding_b is not None:
        description_similarity = rescale_cosine(cosine(description_embedding_a, description_embedding_b))
    else:
        description_similarity = float("nan")
    role_score = role_match_score(concept_a.role, concept_b.role)
    available = [label_similarity, role_score]
    if not np.isnan(description_similarity):
        available.append(description_similarity)
    if llm_judge_score is not None and not np.isnan(llm_judge_score):
        available.append(float(llm_judge_score))

    return {
        "concept_a_id": concept_a.id,
        "concept_a_label": concept_a.label,
        "concept_a_role": concept_a.role,
        "concept_b_id": concept_b.id,
        "concept_b_label": concept_b.label,
        "concept_b_role": concept_b.role,
        "label_similarity": label_similarity,
        "description_similarity": description_similarity,
        "role_match_score": role_score,
        "llm_judge_score": float(llm_judge_score) if llm_judge_score is not None else float("nan"),
        "final_plausibility_score": float(np.mean(available)),
    }


def judge_mapping_with_llm(
    endpoint: str,
    model: str,
    api_key: str,
    concept_a: Concept,
    concept_b: Concept,
    timeout: float = 20.0,
) -> float:
    prompt = (
        "Does concept A play a structurally similar role to concept B in their respective domains? "
        "Score 0.0 to 1.0. Return JSON only with key score.\n\n"
        f"Concept A: {concept_a.label}\nDescription A: {concept_a.description}\nRole A: {concept_a.role}\n"
        f"Concept B: {concept_b.label}\nDescription B: {concept_b.description}\nRole B: {concept_b.role}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'EMPTY'}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))
    content = raw["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return float(parsed["score"])
