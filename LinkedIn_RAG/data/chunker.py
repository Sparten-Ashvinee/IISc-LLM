"""Profile-aware chunking for LinkedIn job postings."""

import re
import hashlib
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    chunk_type: str  # "full", "responsibilities", "requirements", "qualifications", "benefits"
    metadata: dict = field(default_factory=dict)


# Section header patterns commonly found in job descriptions
SECTION_PATTERNS = [
    (r"(?i)(responsibilities|what you.?ll do|your role|the role)", "responsibilities"),
    (r"(?i)(requirements|qualifications|what we.?re looking for|must have|minimum qualifications)", "requirements"),
    (r"(?i)(preferred|nice to have|bonus|desired)", "preferred"),
    (r"(?i)(benefits|perks|what we offer|compensation)", "benefits"),
    (r"(?i)(about us|about the company|who we are)", "about_company"),
]


def _make_chunk_id(text: str, job_id: str, chunk_type: str) -> str:
    """Deterministic chunk ID from content."""
    raw = f"{job_id}:{chunk_type}:{text[:100]}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _split_into_sections(description: str) -> list[tuple[str, str]]:
    """Split a job description into labeled sections."""
    # Find all section boundaries
    boundaries = []
    for pattern, label in SECTION_PATTERNS:
        for m in re.finditer(pattern, description):
            boundaries.append((m.start(), label))

    if not boundaries:
        return [("full", description)]

    boundaries.sort(key=lambda x: x[0])
    sections = []

    # Text before first section header
    if boundaries[0][0] > 50:
        sections.append(("overview", description[:boundaries[0][0]].strip()))

    # Each section
    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(description)
        section_text = description[start:end].strip()
        if len(section_text) > 20:
            sections.append((label, section_text))

    return sections


def chunk_job_posting(row: dict, max_chunk_chars: int = 1000) -> list[Chunk]:
    """Chunk a single job posting into semantic sections.

    If a section is short enough, keep it whole.
    If too long, split on paragraph boundaries.
    """
    job_id = str(row.get("job_id", "unknown"))
    description = row.get("description", "")

    metadata = {
        "job_id": job_id,
        "title": row.get("title", ""),
        "company_name": row.get("company_name", ""),
        "location": row.get("location", ""),
        "experience_level": row.get("formatted_experience_level", ""),
        "skills": row.get("skills_desc", ""),
    }

    sections = _split_into_sections(description)
    chunks = []

    for chunk_type, text in sections:
        if len(text) <= max_chunk_chars:
            chunk_id = _make_chunk_id(text, job_id, chunk_type)
            # Prepend title + company for retrieval context
            enriched = f"{metadata['title']} at {metadata['company_name']}\n{text}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=enriched,
                chunk_type=chunk_type,
                metadata=metadata,
            ))
        else:
            # Split long sections on paragraph boundaries
            paragraphs = text.split("\n\n")
            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) > max_chunk_chars and buffer:
                    cid = _make_chunk_id(buffer, job_id, chunk_type)
                    enriched = f"{metadata['title']} at {metadata['company_name']}\n{buffer}"
                    chunks.append(Chunk(chunk_id=cid, text=enriched,
                                        chunk_type=chunk_type, metadata=metadata))
                    buffer = para
                else:
                    buffer = f"{buffer}\n\n{para}".strip() if buffer else para
            if buffer:
                cid = _make_chunk_id(buffer, job_id, chunk_type)
                enriched = f"{metadata['title']} at {metadata['company_name']}\n{buffer}"
                chunks.append(Chunk(chunk_id=cid, text=enriched,
                                    chunk_type=chunk_type, metadata=metadata))

    # Fallback: if no chunks, use full description
    if not chunks:
        cid = _make_chunk_id(description, job_id, "full")
        enriched = f"{metadata['title']} at {metadata['company_name']}\n{description}"
        chunks.append(Chunk(chunk_id=cid, text=enriched,
                            chunk_type="full", metadata=metadata))

    return chunks


def chunk_dataframe(df, max_chunk_chars: int = 1000) -> list[Chunk]:
    """Chunk all rows in a DataFrame."""
    all_chunks = []
    for _, row in df.iterrows():
        all_chunks.extend(chunk_job_posting(row.to_dict(), max_chunk_chars))
    print(f"Created {len(all_chunks):,} chunks from {len(df):,} postings")
    return all_chunks


if __name__ == "__main__":
    from load_data import load_linkedin_data
    df = load_linkedin_data(max_records=100)
    chunks = chunk_dataframe(df)
    print(f"\nSample chunk:\n{chunks[0].text[:300]}")
    print(f"Type: {chunks[0].chunk_type}")
    print(f"Metadata: {chunks[0].metadata}")
