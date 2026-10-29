import os
import re
import json
import torch
from typing import Optional, List, Dict, Any
from PIL import Image
from pydantic import BaseModel, Field

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


class ESGMetric(BaseModel):
    metric_name: str
    fiscal_year: Optional[int] = None
    value: Optional[float] = None
    unit: Optional[str] = "MT CO2e"
    source_element: Optional[str] = "table"


class PageExtractionResult(BaseModel):
    company_name: str
    report_year: int
    metrics: List[ESGMetric] = Field(default_factory=list)
    qualitative_commitments: List[str] = Field(default_factory=list)
    visual_context_notes: Optional[str] = ""


EXTRACTION_PROMPT = """You are an expert ESG financial auditor analyzing a sustainability report page for {company} (Report Year: {year}).
{page_text_context}
Task: Extract all numerical greenhouse gas (GHG) emissions metrics for Scope 1, Scope 2, and Scope 3 from the visual table and text.
If the table shows multiple historical years (for example 2020, 2021, 2022, 2023, 2024), extract a separate metric item for EACH year shown.

Return ONLY valid JSON in this exact structure:
```json
{{
  "company_name": "{company}",
  "report_year": {year},
  "metrics": [
    {{
      "metric_name": "Scope 1 direct emissions",
      "fiscal_year": {year},
      "value": 100.0,
      "unit": "MT CO2e",
      "source_element": "table"
    }}
  ],
  "qualitative_commitments": [
    "Key emissions target or commitment statement from the page"
  ],
  "visual_context_notes": "Table summary"
}}
```
"""

def extract_metrics_from_page_text(page_text: str, company: str, report_year: int) -> List[Dict[str, Any]]:
    """
    Deterministic multi-year table parser that extracts Scope 1, 2, and 3 figures
    directly from the PDF page text layer across multi-line table cells.
    """
    if not page_text or not page_text.strip():
        return []

    results: List[Dict[str, Any]] = []
    text_clean = page_text.replace("\u2013", "-").replace("\u2014", "-")
    lines = [ln.strip() for ln in text_clean.splitlines() if ln.strip()]
    full_lower = text_clean.lower()

    default_unit = "MT CO2e"
    if any(k in full_lower for k in ["million metric tons", "mmt co2e", "in millions", "million tonnes"]):
        default_unit = "million metric tons CO2e"
    elif any(k in full_lower for k in ["thousand metric tons", "kt co2e", "in thousands", "kilotonnes"]):
        default_unit = "thousand metric tons CO2e"

    # Detect multi-year table column headers anywhere on the page
    header_years: List[int] = []
    for i in range(len(lines)):
        window_header = " ".join(lines[i : min(len(lines), i + 4)])
        yr_matches = re.findall(r"\b(2019|2020|2021|2022|2023|2024)\b", window_header)
        if len(yr_matches) >= 2:
            seen = []
            for y in yr_matches:
                yi = int(y)
                if yi <= report_year and yi not in seen:
                    seen.append(yi)
            if len(seen) >= len(header_years):
                header_years = seen

    scope_rules = [
        ("Scope 1 Direct GHG", [
            r"(?:scope\s*1\b|direct\s+emissions|fossil\s+fuels)",
        ]),
        ("Scope 2 Indirect GHG", [
            r"(?:scope\s*2\b|purchased\s+electricity|market[\-\s]based|location[\-\s]based)",
        ]),
        ("Scope 3 Value Chain GHG", [
            r"(?:scope\s*3\b|value\s+chain\s+emissions|other\s+indirect\s+emissions)",
        ]),
    ]

    # Join 8 adjacent lines per window so multi-line PDF table cells stay grouped together
    windows = []
    for idx in range(len(lines)):
        win = " ".join(lines[idx : min(len(lines), idx + 8)])
        windows.append(win)

    num_regex = re.compile(r"(?<![a-zA-Z0-9_\.\-])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{2,9})(?![a-zA-Z0-9_\.%])")

    for canonical_name, patterns in scope_rules:
        found_for_scope = False
        for win in windows:
            if found_for_scope:
                break
            for pat in patterns:
                m = re.search(pat, win, re.IGNORECASE)
                if not m:
                    continue

                tail = win[m.end() :]
                raw_nums = num_regex.findall(tail)

                valid_floats = []
                for rn in raw_nums:
                    clean_n = rn.replace(",", "")
                    try:
                        v = float(clean_n)
                    except ValueError:
                        continue
                    # Exclude calendar years (2015-2035) and single-digit footnotes
                    if 2015 <= v <= 2035 and "." not in rn:
                        continue
                    if v <= 1.0 and "." not in rn:
                        continue
                    valid_floats.append(v)

                if not valid_floats:
                    continue

                row_lower = win.lower()
                unit = default_unit
                if "million" in row_lower or "mmt" in row_lower:
                    unit = "million metric tons CO2e"
                elif "thousand" in row_lower or "kt" in row_lower:
                    unit = "thousand metric tons CO2e"

                if header_years and len(valid_floats) >= len(header_years):
                    for yr, val in zip(header_years, valid_floats[: len(header_years)]):
                        results.append({
                            "metric_name": canonical_name,
                            "fiscal_year": int(yr),
                            "value": val,
                            "unit": unit,
                            "source_element": "table",
                        })
                elif header_years and len(valid_floats) >= 2:
                    recent_years = [y for y in header_years if y <= report_year][-len(valid_floats):]
                    for yr, val in zip(recent_years, valid_floats):
                        results.append({
                            "metric_name": canonical_name,
                            "fiscal_year": int(yr),
                            "value": val,
                            "unit": unit,
                            "source_element": "table",
                        })
                else:
                    results.append({
                        "metric_name": canonical_name,
                        "fiscal_year": int(report_year),
                        "value": valid_floats[0],
                        "unit": unit,
                        "source_element": "table",
                    })
                    if len(valid_floats) >= 2:
                        results.append({
                            "metric_name": canonical_name,
                            "fiscal_year": int(report_year - 1),
                            "value": valid_floats[1],
                            "unit": unit,
                            "source_element": "table",
                        })

                found_for_scope = True
                break

    return results

def canonicalize_scope_name(metric_name: Any) -> str:
    """Normalize common GHG scope labels to the names used by this module."""
    name = str(metric_name or "").strip()
    normalized = name.lower()

    if re.search(r"\bscope\s*[-:]?\s*1\b", normalized) or "direct emission" in normalized:
        return "Scope 1 Direct GHG"
    if re.search(r"\bscope\s*[-:]?\s*2\b", normalized) or "purchased electricity" in normalized:
        return "Scope 2 Indirect GHG"
    if re.search(r"\bscope\s*[-:]?\s*3\b", normalized) or "value chain" in normalized:
        return "Scope 3 Value Chain GHG"

    return name or "Scope 1 Direct GHG"


def safe_parse_vlm_json(raw_text: str, fallback_year: int) -> Dict[str, Any]:
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    text = fence_match.group(1) if fence_match else raw_text.strip()

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]
    elif start_idx != -1:
        text = text[start_idx:]

    text = re.sub(r"\{\{+", "{", text)
    text = re.sub(r"\}\}+", "}", text)
    text = re.sub(r"//.*?\n", "\n", text)
    text = re.sub(r",\s*([\}\]])", r"\1", text)

    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_brackets > 0 or open_braces > 0:
        last_comma = text.rfind(",")
        if last_comma != -1:
            text = text[:last_comma]
        text += "]" * max(0, text.count("[") - text.count("]"))
        text += "}" * max(0, text.count("{") - text.count("}"))

    try:
        data = json.loads(text)
        valid_metrics = []
        for m in data.get("metrics", []):
            val = m.get("value")
            if val is not None:
                try:
                    vf = float(str(val).replace(",", "").strip())
                    if vf > 0.0 and vf != 100.0:
                        m["value"] = vf
                        m["metric_name"] = canonicalize_scope_name(m.get("metric_name", "Scope 1 Direct GHG"))
                        valid_metrics.append(m)
                except (ValueError, TypeError):
                    continue
        if valid_metrics:
            data["metrics"] = valid_metrics
            return data
    except Exception:
        pass

    extracted_metrics = []
    scope_patterns = [
        ("Scope 1 Direct GHG", r"scope\s*1[^\d\n]*?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{2,9})"),
        ("Scope 2 Indirect GHG", r"scope\s*2[^\d\n]*?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{2,9})"),
        ("Scope 3 Value Chain GHG", r"scope\s*3[^\d\n]*?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{2,9})"),
    ]

    for scope_name, pattern in scope_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(",", "").rstrip(".")
            try:
                val = float(num_str)
                if val > 0.0 and val != 100.0 and not (2018 <= val <= 2030):
                    extracted_metrics.append({
                        "metric_name": scope_name,
                        "fiscal_year": fallback_year,
                        "value": val,
                        "unit": "million metric tons CO2e" if val < 500 else "MT CO2e",
                        "source_element": "table",
                    })
            except ValueError:
                continue

    return {"metrics": extracted_metrics}

class LlamaVisionESGExtractor:
    _model = None
    _processor = None


def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model_name = model_name

    if LlamaVisionESGExtractor._model is None:
        print(f"[+] Loading Optimized Vision Model ({model_name}) into {self.device}...")
        LlamaVisionESGExtractor._processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,
        )
        LlamaVisionESGExtractor._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        ).eval()

    self.model = LlamaVisionESGExtractor._model
    self.processor = LlamaVisionESGExtractor._processor

def extract_metrics_from_image(
    self,
    image: Image.Image,
    company_hint: str,
    year_hint: int,
    page_text: Optional[str] = None,
) -> PageExtractionResult:
    if image is None:
        return PageExtractionResult(
            company_name=company_hint, report_year=year_hint, metrics=[]
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    text_ctx = ""
    if page_text and page_text.strip():
        snippet = page_text.strip()[:2500]
        text_ctx = f"\nReference OCR / Table Text on this Page:\n\"\"\"\n{snippet}\n\"\"\"\n"

    prompt = EXTRACTION_PROMPT.format(
        company=company_hint, year=year_hint, page_text_context=text_ctx
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    raw_text = ""
    try:
        if process_vision_info is not None:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        raw_text = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    except Exception as exc:
        print(f"       [!] VLM inference warning: {exc}")

    data = safe_parse_vlm_json(raw_text, fallback_year=year_hint)
    vlm_metrics_raw = data.get("metrics", []) if isinstance(data, dict) else []

    text_metrics_raw = (
        extract_metrics_from_page_text(page_text, company_hint, year_hint)
        if page_text
        else []
    )

    combined_map: Dict[tuple, ESGMetric] = {}

    for source_list in (vlm_metrics_raw, text_metrics_raw):
        for item in source_list:
            if not isinstance(item, dict):
                continue
            raw_val = item.get("value")
            if raw_val is None:
                continue
            try:
                val = float(str(raw_val).replace(",", "").strip())
            except (ValueError, TypeError):
                continue

            if val <= 0.0 or val == 100.0:
                continue

            cname = canonicalize_scope_name(item.get("metric_name", "Scope 1 Direct GHG"))
            fy = item.get("fiscal_year") or year_hint
            try:
                fy = int(fy)
            except (ValueError, TypeError):
                fy = year_hint

            if fy < 2018 or fy > year_hint:
                fy = year_hint

            unit_str = str(item.get("unit", "MT CO2e"))
            if val < 500 and "million" not in unit_str.lower() and "mmt" not in unit_str.lower():
                unit_str = "million metric tons CO2e"

            key = (cname, fy)
            if key not in combined_map:
                combined_map[key] = ESGMetric(
                    metric_name=cname,
                    fiscal_year=fy,
                    value=val,
                    unit=unit_str,
                    source_element=str(item.get("source_element", "table")),
                )

    parsed_metrics = list(combined_map.values())
    print(f"       [✓] Extracted {len(parsed_metrics)} verified metric(s) from page.")

    return PageExtractionResult(
        company_name=company_hint,
        report_year=year_hint,
        metrics=parsed_metrics,
        qualitative_commitments=data.get("qualitative_commitments", []) if isinstance(data, dict) else [],
        visual_context_notes=data.get("visual_context_notes", "") if isinstance(data, dict) else "",
    )


LlamaVisionESGExtractor.__init__ = __init__
LlamaVisionESGExtractor.extract_metrics_from_image = extract_metrics_from_image