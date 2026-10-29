import json
import re
import torch
from PIL import Image
from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ChartAxisData(BaseModel):
    is_chart_present: bool = Field(..., description="True if a bar or line chart is visible on page")
    chart_title: Optional[str] = Field(None, description="Title of the chart")
    y_axis_min_value: Optional[float] = Field(None, description="The lowest numerical label at the bottom of the y-axis")
    y_axis_max_value: Optional[float] = Field(None, description="The highest numerical label on the y-axis")
    apparent_visual_reduction_pct: Optional[float] = Field(
        None, description="Approximate visual percentage drop of the trend/bars from start to end (0 to 100)"
    )
    reported_numerical_reduction_pct: Optional[float] = Field(
        None, description="Actual numerical percentage drop calculated from printed data points (0 to 100)"
    )


class VNDResult(BaseModel):
    company: str
    fiscal_year: int
    page_number: int
    is_distorted: bool
    dissonance_factor: float
    y_axis_origin: Optional[float]
    explanation: str


class VisualNumericalDissonanceDetector:
    def __init__(self, extractor_instance=None):
        """Re-uses the existing Qwen2-VL-2B instance from LlamaVisionESGExtractor to save VRAM."""
        self.extractor = extractor_instance

    def analyze_page_chart(
        self, 
        image: Image.Image, 
        company: str, 
        fiscal_year: int, 
        page_number: int
    ) -> VNDResult:
        prompt = (
            f"Examine this document page from {company} ({fiscal_year}).\n"
            f"Determine if there is a bar chart or line chart displaying emissions, energy, or environmental progress.\n"
            f"Carefully inspect the vertical (Y) axis:\n"
            f"1. What is the lowest numerical value displayed at the base of the Y-axis? (Does it start at 0, or is it truncated?)\n"
            f"2. Does the visual drop in the chart appear dramatically larger than the actual numbers indicate?\n"
            f"Return ONLY valid JSON matching this schema:\n"
            f"```json\n"
            f'{{\n'
            f'  "is_chart_present": true,\n'
            f'  "chart_title": "Scope 1 & 2 Emissions Trend",\n'
            f'  "y_axis_min_value": 80.0,\n'
            f'  "y_axis_max_value": 120.0,\n'
            f'  "apparent_visual_reduction_pct": 75.0,\n'
            f'  "reported_numerical_reduction_pct": 12.5\n'
            f'}}\n'
            f"```"
        )

        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]

        text_input = self.extractor.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.extractor.processor(images=image, text=text_input, return_tensors="pt").to(self.extractor.device)

        with torch.no_grad():
            output_tokens = self.extractor.model.generate(
                **inputs, 
                max_new_tokens=400,
                temperature=0.1
            )

        output_text = self.extractor.processor.decode(
            output_tokens[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        del inputs, output_tokens

        return self._evaluate_dissonance(output_text, company, fiscal_year, page_number)

    def _evaluate_dissonance(self, text: str, company: str, fiscal_year: int, page_num: int) -> VNDResult:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        raw = match.group(1) if match else text.strip()
        
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]

        try:
            data = json.loads(raw)
            parsed = ChartAxisData(**data)
        except Exception:
            return VNDResult(
                company=company,
                fiscal_year=fiscal_year,
                page_number=page_num,
                is_distorted=False,
                dissonance_factor=1.0,
                y_axis_origin=0.0,
                explanation="No verifiable chart detected or parse failure."
            )

        if not parsed.is_chart_present or parsed.y_axis_min_value is None:
            return VNDResult(
                company=company,
                fiscal_year=fiscal_year,
                page_number=page_num,
                is_distorted=False,
                dissonance_factor=1.0,
                y_axis_origin=parsed.y_axis_min_value,
                explanation="No chart with measurable axes on this page."
            )

        # Baseline origin test
        origin = parsed.y_axis_min_value
        vis_pct = parsed.apparent_visual_reduction_pct or 0.0
        num_pct = parsed.reported_numerical_reduction_pct or 0.0

        # Calculate Dissonance Factor
        if num_pct > 0.0:
            df = round(vis_pct / num_pct, 2)
        else:
            df = 1.0

        # Truncated axis condition: origin > 0 and visual exaggeration > 1.5x
        is_distorted = (origin > 0.0) and (df >= 1.5)

        if is_distorted:
            explanation = (
                f"Truncated Y-axis detected (starts at {origin} instead of 0.0). "
                f"Visual drop appears to be {vis_pct}%, while actual numeric change is only {num_pct}%. "
                f"Dissonance Factor: {df}x."
            )
        else:
            explanation = f"Chart axis grounded honestly at {origin} (DF = {df}x)."

        return VNDResult(
            company=company,
            fiscal_year=fiscal_year,
            page_number=page_num,
            is_distorted=is_distorted,
            dissonance_factor=df,
            y_axis_origin=origin,
            explanation=explanation
        )