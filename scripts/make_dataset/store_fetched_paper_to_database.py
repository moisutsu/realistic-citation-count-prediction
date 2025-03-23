import json
import sqlite3
from pathlib import Path

from classopt import classopt
from tqdm import tqdm


@classopt(default_long=True)
class Opt:
    input_path: Path
    database_path: Path


def line_to_flatten_dict(
    line: str,
    keys_to_dump: list[str] = ["externalIds", "authors", "citations", "references"],
) -> dict:
    flatten_dict = json.loads(line)

    for key in keys_to_dump:
        if key in flatten_dict:
            flatten_dict[key] = json.dumps(flatten_dict[key])

    return flatten_dict


def main(opt: Opt):
    con = sqlite3.connect(opt.database_path)
    cur = con.cursor()

    table_name = "s2"
    index_name = "idIndex"
    index_column = "paperId"
    columns = [
        "paperId",
        "externalIds",
        "title",
        "abstract",
        "citationCount",
        "publicationDate",
        "authors",
        "citations",
        "'references'",
    ]

    columns_with_type = ", ".join(
        [
            f"{column} TEXT" if column != "paperId" else f"{column} TEXT PRIMARY KEY"
            for column in columns
        ]
    )

    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name}({columns_with_type})"
    index_sql = (
        f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({index_column})"
    )

    cur.execute(create_sql)
    cur.execute(index_sql)

    replace_sql = f"REPLACE INTO {table_name}({', '.join(columns)}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"

    keys = [
        "paperId",
        "externalIds",
        "title",
        "abstract",
        "citationCount",
        "publicationDate",
        "authors",
        "citations",
        "references",
    ]
    with opt.input_path.open("r") as f:
        for line in tqdm(list(f), dynamic_ncols=True):
            flatten_dict = line_to_flatten_dict(line)

            values = [
                flatten_dict[key] if key in flatten_dict else None for key in keys
            ]

            cur.execute(replace_sql, values)

    con.commit()

    cur.close()
    con.close()


if __name__ == "__main__":
    opt = Opt.from_args()
    main(opt)
