import sys

sys.path.append(".")

import json
import os
from pathlib import Path

from classopt import classopt, config
from dotenv import load_dotenv
from tqdm import tqdm

from create_ccp_dataset.semantic_scholar_api import featch_paper_details

load_dotenv(verbose=True)


@classopt(default_long=True)
class Opt:
    ids_path: Path
    output_path: Path
    prefix: str = config(default="")
    api_key: str = config(default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))


def main(opt: Opt):
    opt.output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "externalIds",
        "title",
        "abstract",
        "citationCount",
        "publicationDate",
        "authors.name",
        "authors.affiliations",
        "citations.publicationDate",
        "references.publicationDate",
    ]

    fetching_ids = opt.ids_path.read_text().strip().split("\n")
    outputs = []

    for fetching_id in tqdm(fetching_ids):
        try:
            paper = featch_paper_details(
                paper_id=fetching_id,
                fields=fields,
                paper_id_prefix=opt.prefix,
                api_key=opt.api_key,
            )
            outputs.append(paper)
        except Exception:
            print(fetching_id, flush=True)

    opt.output_path.write_text(
        "".join([f"{json.dumps(output)}\n" for output in outputs])
    )


if __name__ == "__main__":
    opt = Opt.from_args()
    main(opt)
