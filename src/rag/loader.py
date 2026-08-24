"""
loader.py

Responsible only for reading markdown files from disk.

Its only job is:
path -> raw text
"""

import os
def load_file(path: str) -> tuple[str, str]:
    """
    Reads a single markdown file.

    Returns:
        (
            source_file_name,
            raw_file_content
        )
    """

    source_file = os.path.basename(path)

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return source_file, raw_text


def load_directory(directory: str) -> list[tuple[str, str]]:
    """
    Reads every markdown file inside a directory.

    Returns:
        [
            ("returns-policy.md", "<raw markdown>"),
            ("refund-policy.md", "<raw markdown>")
        ]
    """

    files = []

    for filename in sorted(os.listdir(directory)):

        if not filename.endswith(".md"):
            continue

        path = os.path.join(directory, filename)

        try:
            files.append(load_file(path))

        except OSError as e:
            print(
                f"[loader] WARNING: failed reading {filename}: {e}"
            )

    return files

if __name__ == "__main__":

    documents = load_directory("../../ai-agent-intern-test/knowledge-base")

    print(f"Loaded {len(documents)} files\n")

    for filename, raw_text in documents:
        print(
            f"{filename}: {len(raw_text)} characters"
        )