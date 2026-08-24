"""
parser.py

Responsible for converting raw markdown text into
structured ParsedDocument objects.

It handles:
- YAML front matter extraction
- metadata validation

It does NOT:
- read files
- scan directories
- chunk documents
"""

from dataclasses import dataclass
import yaml


@dataclass
class ParsedDocument:
    source_file: str
    metadata: dict
    body: str



def split_front_matter(
    raw_text: str,
    source_file: str
) -> tuple[dict, str]:

    lines = raw_text.splitlines()

    if not lines or lines[0].strip() != "---":

        raise ValueError(
            f"{source_file}: missing YAML front matter"
        )

    closing_index = None

    for i in range(1, len(lines)):

        if lines[i].strip() == "---":
            closing_index = i
            break


    if closing_index is None:

        raise ValueError(
            f"{source_file}: missing closing front matter delimiter"
        )


    front_matter_text = "\n".join(
        lines[1:closing_index]
    )

    body_text = "\n".join(
        lines[closing_index + 1:]
    ).strip()


    metadata = yaml.safe_load(front_matter_text) or {}


    return metadata, body_text



REQUIRED_METADATA_FIELDS = [
    "document_id",
    "title",
    "status",
    "policy_authority",
]



def validate_metadata(
    metadata: dict,
    source_file: str
):

    missing = [
        field
        for field in REQUIRED_METADATA_FIELDS
        if field not in metadata
    ]


    if missing:

        raise ValueError(
            f"{source_file}: missing fields {missing}"
        )



def parse_document(
    source_file: str,
    raw_text: str
) -> ParsedDocument:


    metadata, body = split_front_matter(
        raw_text,
        source_file
    )


    validate_metadata(
        metadata,
        source_file
    )


    return ParsedDocument(
        source_file=source_file,
        metadata=metadata,
        body=body
    )

if __name__ == "__main__":

    from loader import load_directory


    raw_documents = load_directory(
        "../../ai-agent-intern-test/knowledge-base"
    )


    parsed_documents = []


    for filename, raw_text in raw_documents:

        try:

            doc = parse_document(
                filename,
                raw_text
            )

            parsed_documents.append(doc)


        except ValueError as e:

            print(
                f"[parser] WARNING: {e}"
            )


    print(
        f"\nParsed {len(parsed_documents)} documents\n"
    )


    for doc in parsed_documents:

        print(
            f"""
                File: {doc.source_file}
                ID: {doc.metadata.get("document_id")}
                Title: {doc.metadata.get("title")}
                Status: {doc.metadata.get("status")}
                Authority: {doc.metadata.get("policy_authority")}
                Body chars: {len(doc.body)}
                -----------------------------
            """
        )