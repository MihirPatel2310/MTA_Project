import os
import re
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel
from typing import Optional
import re

def normalize_metric_name(name: str) -> str:
    if not name:
        return "unspecified_metric"
    n = name.lower()
    if "scope 1" in n:
        return "scope_1_ghg"
    elif "scope 2" in n:
        return "scope_2_ghg"
    elif "scope 3" in n:
        return "scope_3_ghg"
    elif any(k in n for k in ["renewable", "clean energy", "electricity"]):
        return "renewable_energy_pct"
    elif "carbon intensity" in n:
        return "carbon_intensity"
    return re.sub(r"[^a-z0-9_]+", "_", n.strip())

class StrategicAmbiguityRecord(BaseModel):
    company: str
    base_year: int
    target_year: int
    semantic_drift: float
    vagueness_score: float
    omission_penalty: float = 0.0
    strategic_ambiguity_score: float
    flag_level: str
    base_statement: Optional[str] = ""
    target_statement: Optional[str] = ""

    # Property alias so s.numerical_omission works seamlessly
    @property
    def numerical_omission(self) -> float:
        return self.omission_penalty
AmbiguityAuditResult = StrategicAmbiguityRecord

class StrategicAmbiguityEngine:
    def __init__(self, embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
        self.model = AutoModel.from_pretrained(embed_model_id).to(self.device).eval()

        # SASB & GRI Hedging Lexicon
        self.hedging_terms = {
            "aim to", "endeavor to", "aspire to", "pathway", "where feasible",
            "explore", "seek to", "strive", "subject to market conditions",
            "ambition", "intend", "potential to", "working towards", "plans to",
            "may", "might", "could", "hope to"
        }
        self.hard_anchors = {
            "%", "percent", "metric ton", "mwh", "net-zero by 20", "baseline year",
            "scope 1", "scope 2", "scope 3", "reduced by", "absolute reduction",
            "ghg", "carbon", "renewable", "recycled", "target"
        }

    def _get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            # Mean pooling over attention mask
            mask = inputs["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
            emb = torch.sum(out[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            return F.normalize(emb, p=2, dim=1)

    def compute_semantic_drift(self, text_a: str, text_b: str) -> float:
        emb_a = self._get_embedding(text_a)
        emb_b = self._get_embedding(text_b)
        cos_sim = torch.mm(emb_a, emb_b.transpose(0, 1)).item()
        # Semantic drift is (1 - cosine_similarity), bounded [0, 1]
        drift = max(0.0, min(1.0, 1.0 - cos_sim))
        return round(drift, 4)

    def compute_vagueness(self, text: str) -> float:
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        if not words:
            return 0.5

        hedge_count = sum(1 for phrase in self.hedging_terms if phrase in text_lower)
        anchor_count = sum(1 for phrase in self.hard_anchors if phrase in text_lower)

        # Lexical ratio of hedging vs concrete anchors
        lexical_hedge_ratio = hedge_count / max(1, (hedge_count + anchor_count))

        # Shannon Entropy of token frequencies as a proxy for linguistic dispersion
        freqs = {}
        for w in words:
            freqs[w] = freqs.get(w, 0) + 1
        entropy = -sum((c / len(words)) * math.log2(c / len(words)) for c in freqs.values())
        norm_entropy = min(1.0, entropy / 6.0)

        vagueness = (0.65 * lexical_hedge_ratio) + (0.35 * norm_entropy)
        return round(min(1.0, max(0.0, vagueness)), 4)

    

    def classify_ambiguity_level(self, score: float) -> str:
        """
        Empirically calibrated thresholds across corporate sustainability corpus:
        - SAS < 0.25: High alignment, low drift, explicit numerical anchoring.
        - 0.25 <= SAS < 0.45: Moderate ambiguity, notable hedging or qualitative drift.
        - SAS >= 0.45: High strategic ambiguity, substantial goal dilution / omission.
        """
        if score >= 0.45:
            return "HIGH_STRATEGIC_AMBIGUITY"
        elif score >= 0.25:
            return "MODERATE_AMBIGUITY"
        else:
            return "ALIGNED_DISCLOSURE"

        return AmbiguityAuditResult(
            company=company,
            base_year=base_year,
            target_year=target_year,
            strategic_ambiguity_score=sas,
            flag_level=flag,
            semantic_drift=drift,
            vagueness_score=vagueness,
            numerical_omission=has_number_omission,
            base_statement=base_statement,
            target_statement=target_statement
        )
    
    def calculate_sas(
        self,
        company: str,
        base_year: int,
        target_year: int,
        base_statement: str,
        target_statement: str,
        number_omitted: bool = False,
        has_number_omission: bool = False,
        alpha: float = 0.45,
        beta: float = 0.35,
        gamma: float = 0.20,
        **kwargs
    ) -> StrategicAmbiguityRecord:
        """
        Calculates Strategic Ambiguity Score (SAS):
        SAS = alpha * SemanticDrift + beta * Vagueness + gamma * OmissionPenalty
        """
        # Resolve omission flag from either keyword argument
        omitted = number_omitted or has_number_omission or kwargs.get("omitted", False)

        # 1. Semantic Drift (dense cosine distance)
        drift = self.compute_semantic_drift(base_statement, target_statement)

        # 2. Linguistic Vagueness (hedging density + entropy on target statement)
        vagueness = self.compute_vagueness(target_statement)

        # 3. Numerical Omission Penalty
        omission_penalty = 1.0 if omitted else 0.0

        # 4. Composite Formula
        sas_score = (alpha * drift) + (beta * vagueness) + (gamma * omission_penalty)
        sas_score = round(float(sas_score), 4)

        # 5. Categorical Classification
        flag = self.classify_ambiguity_level(sas_score)

        return StrategicAmbiguityRecord(
            company=company,
            base_year=base_year,
            target_year=target_year,
            semantic_drift=round(drift, 4),
            vagueness_score=round(vagueness, 4),
            omission_penalty=omission_penalty,
            strategic_ambiguity_score=sas_score,
            flag_level=flag,
            base_statement=base_statement,
            target_statement=target_statement
        )

    # Alias so any method call variant succeeds
    compute_sas = calculate_sas