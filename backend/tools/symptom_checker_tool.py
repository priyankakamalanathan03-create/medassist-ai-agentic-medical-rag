"""
tools/symptom_checker_tool.py — Symptom-to-condition matching tool.

Searches the indexed medical docs for text relating to given symptoms
and returns matched conditions with source evidence.
"""
from tools.tool_registry import ToolRegistry, Tool


SYMPTOM_KEYWORDS = {
    "diabetes":        ["blood sugar", "glucose", "insulin", "polyuria", "polydipsia", "hba1c"],
    "asthma":          ["wheezing", "shortness of breath", "bronchial", "airway", "inhaler"],
    "cardiovascular":  ["chest pain", "hypertension", "blood pressure", "heart", "stroke", "lipid"],
    "cancer":          ["tumor", "malignant", "carcinoma", "biopsy", "chemotherapy", "metastasis"],
    "respiratory":     ["cough", "lung", "oxygen", "spirometry", "dyspnea", "bronchitis"],
    "nutrition":       ["diet", "vitamin", "calorie", "nutrient", "obesity", "bmi"],
}


def _check_symptoms(symptoms: str) -> dict:
    """
    Analyse a list of symptoms and return likely conditions with supporting
    evidence from the indexed medical knowledge base.

    Args:
        symptoms: Comma-separated symptom descriptions
                  (e.g. "high blood sugar, frequent urination, fatigue")

    Returns:
        {
            "symptoms_received": str,
            "possible_conditions": [
                {
                    "condition": str,
                    "keyword_match": str,  # heuristic overlap, not clinical confidence
                    "evidence": [str]    # supporting text snippets
                }
            ],
            "disclaimer": str
        }
    """
    from rag.retriever import retriever

    symptom_list = [s.strip().lower() for s in symptoms.split(",") if s.strip()]

    # Score conditions based on keyword overlap
    condition_scores: dict[str, int] = {}
    for condition, keywords in SYMPTOM_KEYWORDS.items():
        score = 0
        for symptom in symptom_list:
            for kw in keywords:
                if kw in symptom or symptom in kw:
                    score += 2
            # Also check free text
            for kw in keywords:
                if any(w in kw for w in symptom.split()):
                    score += 1
        if score > 0:
            condition_scores[condition] = score

    if not condition_scores:
        # Fall back to document search
        docs     = retriever.retrieve(symptoms, k=5)
        evidence = [f"[{d['source']}, p.{d['page']}] {d['text'][:200]}..." for d in docs]
        return {
            "symptoms_received":    symptoms,
            "possible_conditions":  [],
            "document_evidence":    evidence,
            "disclaimer":           _disclaimer(),
        }

    # Build output
    possible = []
    top_conditions = sorted(condition_scores, key=condition_scores.get, reverse=True)[:3]

    for condition in top_conditions:
        score      = condition_scores[condition]
        keyword_match = "High" if score >= 6 else "Medium" if score >= 3 else "Low"

        # Retrieve supporting passages from documents
        docs     = retriever.retrieve(f"{condition} {symptoms}", k=3)
        evidence = [
            f"[{d['source']}, p.{d['page']}] {d['text'][:180]}..."
            for d in docs
        ]

        possible.append({
            "condition":     condition.capitalize(),
            "keyword_match": keyword_match,
            "evidence":      evidence,
        })

    return {
        "symptoms_received":   symptoms,
        "possible_conditions": possible,
        "interpretation_note": (
            "Keyword-match strength is a heuristic text overlap, not a diagnosis or "
            "probability of having a condition."
        ),
        "disclaimer":          _disclaimer(),
    }


def _disclaimer() -> str:
    return (
        "⚠️ This is an AI-assisted symptom analysis based on indexed medical documents only. "
        "It is NOT a clinical diagnosis. Always consult a licensed healthcare professional."
    )


def register_symptom_checker_tool(registry: ToolRegistry) -> None:
    registry.register(Tool(
        name="check_symptoms",
        description=(
            "Match symptom terms against indexed medical topics and return related passages "
            "with citations. Keyword-match strength is not diagnostic confidence or a "
            "probability. Always includes a medical disclaimer."
        ),
        func=_check_symptoms,
        parameters={
            "symptoms": "Comma-separated list of symptoms (e.g. 'chest pain, shortness of breath, fatigue')",
        },
    ))
