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


@classopt(default_long=True, default_short=True)
class Opt:
    ids_path: Path
    output_s2_ids_to_calculate_citation_count: Path = config(short="-os")
    お: Path = config(short="-oi")
    prefix: str = config(default="ARXIV:")
    api_key: str = config(default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))


def main(opt: Opt):
    opt.output_s2_ids_to_calculate_citation_count.parent.mkdir(
        parents=True, exist_ok=True
    )
    opt.output_input_ids_to_s2_ids_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["references"]

    fetching_ids = opt.ids_path.read_text().strip().split("\n")
    output_s2_ids_to_calculate_citation_count = []
    output_input_ids_to_s2_ids = {}

    for fetching_id in tqdm(fetching_ids):
        try:
            paper = featch_paper_details(
                paper_id=fetching_id,
                fields=fields,
                paper_id_prefix=opt.prefix,
                api_key=opt.api_key,
            )
            output_s2_ids_to_calculate_citation_count.append(paper["paperId"])
            output_s2_ids_to_calculate_citation_count.extend(
                [reference["paperId"] for reference in paper["references"]]
            )

            output_input_ids_to_s2_ids[fetching_id] = paper["paperId"]

        except Exception:
            print(fetching_id, flush=True)

    output_s2_ids_to_calculate_citation_count = list(
        set(output_s2_ids_to_calculate_citation_count)
    )
    opt.output_s2_ids_to_calculate_citation_count.write_text(
        "".join(
            [
                f"{output_id}\n"
                for output_id in output_s2_ids_to_calculate_citation_count
            ]
        )
    )

    opt.output_input_ids_to_s2_ids_path.write_text(
        json.dumps(output_input_ids_to_s2_ids, indent=2)
    )


if __name__ == "__main__":
    opt = Opt.from_args()
    main(opt)
