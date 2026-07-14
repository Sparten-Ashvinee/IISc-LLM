"""Grounded answer generation with mandatory citations."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = """You are a professional recruiter assistant. Answer the user's query using ONLY the provided job posting context.

Rules:
1. Every factual claim MUST cite the source using [Source N] format.
2. If the context doesn't contain enough information, say "I don't have enough information to answer that."
3. Never invent job details, companies, or requirements not present in the context.
4. Be concise and structured — use bullet points for multiple results.
"""

CONTEXT_TEMPLATE = """
--- Source {idx} ---
Job: {title} at {company}
Location: {location}
{text}
"""


class Generator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 quantization: str = "4bit", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Quantization config
        quant_config = None
        if quantization == "4bit" and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantization == "8bit" and self.device == "cuda":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, query: str, retrieved_chunks: list[dict],
                 max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate a grounded answer with citations."""

        # Build context from retrieved chunks
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk.get("metadata", {})
            context_parts.append(CONTEXT_TEMPLATE.format(
                idx=idx,
                title=meta.get("title", "N/A"),
                company=meta.get("company_name", "N/A"),
                location=meta.get("location", "N/A"),
                text=chunk["text"][:800],  # truncate very long chunks
            ))

        context_str = "\n".join(context_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated portion
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        return answer.strip()
