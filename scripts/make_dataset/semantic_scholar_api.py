import json
from time import sleep
from typing import Optional

import requests

# request rate: 100 req/sec
SLEEP_SECONDS = 0.011


def call_api(url: str, api_key: Optional[str] = None) -> dict:
    headers = None
    if api_key:
        headers = {"x-api-key": api_key}

    response = json.loads(requests.get(url, headers=headers).text)
    sleep(SLEEP_SECONDS)

    return response


def featch_paper_details(
    paper_id: str,
    fields: list[str],
    paper_id_prefix: str = "",
    api_key: Optional[str] = None,
) -> dict:
    END_POINT = (
        "https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields={fields}"
    )

    url = END_POINT.format(
        paper_id=f"{paper_id_prefix}{paper_id}", fields=",".join(fields)
    )

    response = call_api(url, api_key)

    if response.get("citationCount", 0) >= 1000:
        response["citations"] = fetch_all_paper_citations(
            paper_id=paper_id,
            fields=["paperId", "publicationDate"],
            paper_id_prefix=paper_id_prefix,
            api_key=api_key,
        )

    return response


def fetch_all_paper_citations(
    paper_id: str,
    fields: list[str],
    paper_id_prefix: str = "",
    api_key: Optional[str] = None,
) -> list[dict]:
    END_POINT = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields={fields}&offset={offset}&limit={limit}"

    all_citations = []
    offset = 0
    limit = 1000

    # According to API specifications, this process is always completed within 10 times.
    for _ in range(10):
        url = END_POINT.format(
            paper_id=f"{paper_id_prefix}{paper_id}",
            fields=",".join(fields),
            offset=offset,
            limit=limit,
        )
        response = call_api(url, api_key)

        all_citations.extend(response["data"])

        if "next" in response:
            offset = int(response["next"])
        else:
            break

    return all_citations
