import json
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from classopt import classopt, config
from tqdm import tqdm

S2_ID_TO_PAPER = dict()


@classopt(default_long=True)
class Opt:
    paper_ids_path: Path
    convert_to_s2_id_path: Path
    database_path: Path
    output_dir: Path

    oldest_date_for_train: list[int] = config(default=[2016, 4], nargs=2)
    test_date: list[int] = config(default=[2021, 4], nargs=2)
    n_years_after: int = 1
    mode_for_citation_counts_within_n_years_after_publication: str = config(
        default="use_full", choices=["use_full", "not_use", "complement", "no_complement"]
    )

    percent_threshold_for_complement: int = 90
    percent_ranges_for_degree_of_early_adoptions: list[float] = [
        0.1,
        1,
        2.5,
        5,
        10,
        25,
        100,
    ]

    table_name: str = "s2"


@dataclass
class Paper:
    s2_id: str
    external_ids: dict
    title: str
    abstract: str
    citation_count: int
    publication_date: tuple[int, int]
    authors: list[dict]
    citations: list[dict]
    references: list[dict]


def save_as_jsonl(output_path: Path, data: list[dict]):
    output_path.write_text("".join([f"{json.dumps(line)}\n" for line in data]))


def convert_string_date_to_tuple_date(date: str) -> tuple[int, int]:
    year = int(date.split("-")[0])
    month = int(date.split("-")[1])

    return year, month


def extract_date(line: dict) -> tuple[int, int]:
    return convert_string_date_to_tuple_date(line["date"])


def difference_date_as_month(date1: tuple[int, int], date2: tuple[int, int]) -> int:
    return (date1[0] - date2[0]) * 12 + date1[1] - date2[1]


def select_paper(s2_id: str, cur: sqlite3.Cursor) -> Optional[Paper]:
    global S2_ID_TO_PAPER

    if s2_id in S2_ID_TO_PAPER:
        return S2_ID_TO_PAPER[s2_id]

    select_paper_sql = f"SELECT * FROM {opt.table_name} WHERE paperId=?"

    selected_data = cur.execute(select_paper_sql, (s2_id,)).fetchone()
    if selected_data is None:
        return None

    (
        s2_id,
        external_ids,
        title,
        abstract,
        citation_count,
        publication_date,
        authors,
        citations,
        references,
    ) = selected_data

    if publication_date is None:
        return None

    # Convert types
    external_ids = json.loads(external_ids)
    citation_count = int(citation_count)
    publication_date = convert_string_date_to_tuple_date(publication_date)
    authors = json.loads(authors)
    citations = json.loads(citations)
    references = json.loads(references)

    # Remove unnecessary values and convert internal date type
    for i in range(len(citations) - 1, -1, -1):
        if "citingPaper" in citations[i]:
            citations[i] = citations[i]["citingPaper"]

        if citations[i]["paperId"] is None or citations[i]["publicationDate"] is None:
            citations.pop(i)
            continue

        citations[i]["publicationDate"] = convert_string_date_to_tuple_date(
            citations[i]["publicationDate"]
        )

    for i in range(len(references) - 1, -1, -1):
        if references[i]["paperId"] is None or references[i]["publicationDate"] is None:
            references.pop(i)
            continue

        references[i]["publicationDate"] = convert_string_date_to_tuple_date(
            references[i]["publicationDate"]
        )

    paper = Paper(
        s2_id=s2_id,
        external_ids=external_ids,
        title=title,
        abstract=abstract,
        citation_count=citation_count,
        publication_date=publication_date,
        authors=authors,
        citations=citations,
        references=references,
    )

    S2_ID_TO_PAPER[s2_id] = paper

    return paper


def calculate_citation_count_after_m_months(paper: Paper, m_months_after: int) -> int:
    m_months_after_publication_year = (
        paper.publication_date[0]
        + (paper.publication_date[1] + m_months_after - 1) // 12
    )
    m_months_after_publication_month = (
        paper.publication_date[1] + m_months_after - 1
    ) % 12 + 1

    m_months_after_publication = (
        m_months_after_publication_year,
        m_months_after_publication_month,
    )

    return sum(
        [
            1
            for citation in paper.citations
            if paper.publication_date
            <= citation["publicationDate"]
            <= m_months_after_publication
        ]
    )


def calculate_complemented_citation_count_after_m_months(
    paper: Paper,
    m_months_after: int,
    test_date: tuple[int, int],
    elapsed_months_to_citation_counts: dict[int, list[int]],
    elapsed_months_to_mean_citation_count: dict[int, float],
    percent_threshold_for_complement: int,
) -> int:
    elapsed_months = difference_date_as_month(test_date, paper.publication_date)
    citation_count_at_publication_of_test_paper = calculate_citation_count_after_m_months(
        paper, elapsed_months
    )

    threshold_index = int(
        len(elapsed_months_to_citation_counts[elapsed_months])
        * percent_threshold_for_complement
        // 100
    )
    threshold_citation_count = sorted(
        elapsed_months_to_citation_counts[elapsed_months]
    )[threshold_index]

    # ratio-based complement
    if citation_count_at_publication_of_test_paper >= threshold_citation_count:
        complemented_citation_count = citation_count_at_publication_of_test_paper * (
            elapsed_months_to_mean_citation_count[m_months_after]
            / elapsed_months_to_mean_citation_count[elapsed_months]
        )

    # case-based complement
    else:
        complemented_citation_count = statistics.median(
            [
                elapsed_months_to_citation_counts[m_months_after][
                    index_of_same_citation_count_and_elapsed_months
                ]
                for index_of_same_citation_count_and_elapsed_months in range(
                    len(elapsed_months_to_citation_counts[m_months_after])
                )
                if elapsed_months_to_citation_counts[elapsed_months][
                    index_of_same_citation_count_and_elapsed_months
                ]
                == citation_count_at_publication_of_test_paper
            ]
        )

    return round(complemented_citation_count)


def create_elapsed_months_to_citation_counts(
    papers_use_for_train: list[Paper], n_years_after: int
) -> dict[int, list[int]]:
    elapsed_months_to_citation_counts: dict[int, list[int]] = {}

    for elapsed_months in range(1, n_years_after * 12 + 1):
        elapsed_months_to_citation_counts[elapsed_months] = []
        for paper in papers_use_for_train:
            elapsed_months_to_citation_counts[elapsed_months].append(
                calculate_citation_count_after_m_months(paper, elapsed_months)
            )

    return elapsed_months_to_citation_counts


def calculate_degree_of_early_adoptions(
    paper: Paper,
    m_months_after: int,
    elapsed_months_to_citation_counts: dict[int, list[int]],
    percent_ranges_for_degree_of_early_adoptions: list[float],
    cur: sqlite3.Cursor,
) -> list[int]:
    elapsed_months_to_reference_paper_citation_counts_at_publication = dict()
    for elapsed_months in range(1, m_months_after + 1):
        elapsed_months_to_reference_paper_citation_counts_at_publication[
            elapsed_months
        ] = []

    for reference in paper.references:
        if (
            not (
                paper.publication_date[0] - m_months_after // 12,
                paper.publication_date[1],
            )
            <= reference["publicationDate"]
            < paper.publication_date
        ):
            continue

        reference_paper = select_paper(reference["paperId"], cur)

        if reference_paper is None:
            continue

        elapsed_months = difference_date_as_month(
            paper.publication_date, reference["publicationDate"]
        )

        elapsed_months_to_reference_paper_citation_counts_at_publication[
            elapsed_months
        ].append(
            calculate_citation_count_after_m_months(reference_paper, elapsed_months)
        )

    degree_of_early_adoptions = []

    best_degree_of_early_adoption = (
        len(percent_ranges_for_degree_of_early_adoptions) - 1
    )
    current_best_degree_of_early_adoption = -1

    for (
        elapsed_months,
        reference_paper_citation_counts_at_publication,
    ) in elapsed_months_to_reference_paper_citation_counts_at_publication.items():
        thresholds = []
        for percent in percent_ranges_for_degree_of_early_adoptions:
            sorted_citation_counts = sorted(
                elapsed_months_to_citation_counts[elapsed_months], reverse=True
            )
            percent_index = min(
                max(0, int(len(sorted_citation_counts) * (percent / 100))),
                len(sorted_citation_counts) - 1,
            )
            thresholds.append(sorted_citation_counts[percent_index])

        current_elapsed_month_max_citation_count = (
            max(reference_paper_citation_counts_at_publication)
            if len(reference_paper_citation_counts_at_publication) != 0
            else -1
        )
        current_degree_of_early_adoption = -1
        for i, threshold in enumerate(thresholds):
            if current_elapsed_month_max_citation_count > threshold:
                current_degree_of_early_adoption = best_degree_of_early_adoption - i
                break

        current_best_degree_of_early_adoption = max(
            current_best_degree_of_early_adoption, current_degree_of_early_adoption
        )
        degree_of_early_adoptions.append(current_best_degree_of_early_adoption)

    return degree_of_early_adoptions


def main(opt: Opt):
    oldest_date_for_train: tuple[int, int] = tuple(opt.oldest_date_for_train)
    test_date: tuple[int, int] = tuple(opt.test_date)

    con = sqlite3.connect(opt.database_path)
    cur = con.cursor()

    opt.output_dir.mkdir(parents=True, exist_ok=True)

    paper_ids: list[dict] = opt.paper_ids_path.read_text().strip().split("\n")
    convert_to_s2_id: dict[str, str] = json.loads(opt.convert_to_s2_id_path.read_text())

    papers_use_for_dataset_with_paper_ids: list[tuple[Paper, str]] = []
    papers_use_for_train: list[Paper] = []
    for paper_id in tqdm(
        paper_ids, desc="Select papers use for dataset", dynamic_ncols=True
    ):
        s2_id = convert_to_s2_id.get(paper_id, None)

        if s2_id is None:
            continue

        paper = select_paper(s2_id, cur)
        if paper is None:
            continue

        if not oldest_date_for_train <= paper.publication_date <= test_date:
            continue

        papers_use_for_dataset_with_paper_ids.append((paper, paper_id))

        if paper.publication_date != test_date:
            papers_use_for_train.append(paper)

    elapsed_months_to_citation_counts = create_elapsed_months_to_citation_counts(
        papers_use_for_train, opt.n_years_after
    )

    elapsed_months_to_mean_citation_count: dict[int, float] = dict()
    for elapsed_months, citation_counts in elapsed_months_to_citation_counts.items():
        elapsed_months_to_mean_citation_count[elapsed_months] = sum(
            citation_counts
        ) / len(citation_counts)

    train_set: list[dict] = []
    test_set: list[dict] = []

    # make papers_use_for_dataset_with_paper_ids unique
    # see: https://stackoverflow.com/questions/10024646/how-to-get-list-of-objects-with-unique-attribute
    added = set()
    papers_use_for_dataset_with_paper_ids = [
        added.add(line[1]) or line
        for line in papers_use_for_dataset_with_paper_ids
        if line[1] not in added
    ]

    for paper, paper_id in tqdm(
        papers_use_for_dataset_with_paper_ids,
        desc="Creating dataset",
        dynamic_ncols=True,
    ):
        # Papers within n years of publication of the paper for testing
        if (
            (test_date[0] - opt.n_years_after, test_date[1])
            < paper.publication_date
            < test_date
        ):
            if (
                opt.mode_for_citation_counts_within_n_years_after_publication
                == "use_full"
            ):
                citation_count_after_n_years = calculate_citation_count_after_m_months(
                    paper=paper, m_months_after=opt.n_years_after * 12
                )

            elif (
                opt.mode_for_citation_counts_within_n_years_after_publication
                == "not_use"
            ):
                continue

            elif (
                opt.mode_for_citation_counts_within_n_years_after_publication
                == "complement"
            ):
                citation_count_after_n_years = calculate_complemented_citation_count_after_m_months(
                    paper=paper,
                    m_months_after=opt.n_years_after * 12,
                    test_date=test_date,
                    elapsed_months_to_citation_counts=elapsed_months_to_citation_counts,
                    elapsed_months_to_mean_citation_count=elapsed_months_to_mean_citation_count,
                    percent_threshold_for_complement=opt.percent_threshold_for_complement,
                )

            elif (
                opt.mode_for_citation_counts_within_n_years_after_publication
                == "no_complement"
            ):
                elapsed_months = difference_date_as_month(test_date, paper.publication_date)

                if paper.publication_date == test_date or elapsed_months >= opt.n_years_after * 12:
                    citation_count_after_n_years = calculate_citation_count_after_m_months(
                        paper=paper, m_months_after=opt.n_years_after * 12
                    )
                else:
                    citation_count_after_n_years = calculate_citation_count_after_m_months(
                        paper=paper, m_months_after=elapsed_months
                    )

        # Papers for testing or papers for training that are more than n years after publication
        else:
            citation_count_after_n_years = calculate_citation_count_after_m_months(
                paper=paper, m_months_after=opt.n_years_after * 12
            )

        degree_of_early_adoptions = calculate_degree_of_early_adoptions(
            paper=paper,
            m_months_after=opt.n_years_after * 12,
            elapsed_months_to_citation_counts=elapsed_months_to_citation_counts,
            percent_ranges_for_degree_of_early_adoptions=opt.percent_ranges_for_degree_of_early_adoptions,
            cur=cur,
        )

        output = {
            "s2_id": paper.s2_id,
            "publisher": paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "citation_counts": citation_count_after_n_years,
            "elapsed_months": opt.n_years_after * 12,
            "degree_of_early_adoptions": degree_of_early_adoptions,
            "publication_year": paper.publication_date[0],
            "publication_month": paper.publication_date[1],
        }
        if paper.publication_date == test_date:
            test_set.append(output)
        else:
            train_set.append(output)

    save_as_jsonl(opt.output_dir / "train.jsonl", train_set)
    save_as_jsonl(opt.output_dir / "test.jsonl", test_set)

    # Save test set as development set to simplify experimental code
    save_as_jsonl(opt.output_dir / "valid.jsonl", test_set)

    cur.close()
    con.close()


if __name__ == "__main__":
    opt = Opt.from_args()
    main(opt)
