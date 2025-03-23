#!/bin/bash

YEAR_RANGE=5
N_YEARS_AFTER=1
TEST_MONTHS=(4 3 2 1 12 11 10 9 8 7 6 5)
TEST_YEARS=(2021 2021 2021 2021 2020 2020 2020 2020 2020 2020 2020 2020)

OLDEST_TRAIN_YEARS=()
for TEST_YEAR in "${TEST_YEARS[@]}"
do
    OLDEST_TRAIN_YEARS+=("$((TEST_YEAR-YEAR_RANGE))")
done

for (( i=0; i < "${#TEST_YEARS[@]}"; i++ ))
do
    TEST_YEAR="${TEST_YEARS[i]}"
    OLDEST_TRAIN_YEAR="${OLDEST_TRAIN_YEARS[i]}"
    TEST_MONTH="${TEST_MONTHS[i]}"

    # complement
    OUTPUT_DIR="datasets/ccp/biorxiv/${YEAR_RANGE}_years/use_current_citation/test_${TEST_YEAR}-${TEST_MONTH}/${N_YEARS_AFTER}_year_later_citation_complemented"
    echo "$OUTPUT_DIR"
    python create_ccp_dataset/create_dataset_from_db.py \
        --paper_ids_path datasets/paper_ids/biorxiv/2014_1_17-2022_4_30-doi-plant.txt \
        --convert_to_s2_id_path datasets/paper_ids/convert/biorxiv_2014_1_17-2022_4_30-doi-plant_to_s2_ids.json \
        --database_path /local2/hirako/s2.db \
        --output_dir "$OUTPUT_DIR" \
        --oldest_date_for_train "$OLDEST_TRAIN_YEAR" "$TEST_MONTH" \
        --test_date "$TEST_YEAR" "$TEST_MONTH" \
        --n_years_after "$N_YEARS_AFTER" \
        --mode_for_citation_counts_within_n_years_after_publication complement

    # use_current
    OUTPUT_DIR="datasets/ccp/biorxiv/${YEAR_RANGE}_years/use_current_citation/test_${TEST_YEAR}-${TEST_MONTH}/${N_YEARS_AFTER}_year_later_citation"
    echo "$OUTPUT_DIR"
    python create_ccp_dataset/create_dataset_from_db.py \
        --paper_ids_path datasets/paper_ids/biorxiv/2014_1_17-2022_4_30-doi-plant.txt \
        --convert_to_s2_id_path datasets/paper_ids/convert/biorxiv_2014_1_17-2022_4_30-doi-plant_to_s2_ids.json \
        --database_path /local2/hirako/s2.db \
        --output_dir "$OUTPUT_DIR" \
        --oldest_date_for_train "$OLDEST_TRAIN_YEAR" "$TEST_MONTH" \
        --test_date "$TEST_YEAR" "$TEST_MONTH" \
        --n_years_after "$N_YEARS_AFTER" \
        --mode_for_citation_counts_within_n_years_after_publication not_use

    # use_future
    OUTPUT_DIR="datasets/ccp/biorxiv/${YEAR_RANGE}_years/use_future_citation/test_${TEST_YEAR}-${TEST_MONTH}/${N_YEARS_AFTER}_year_later_citation"
    echo "$OUTPUT_DIR"
    python create_ccp_dataset/create_dataset_from_db.py \
        --paper_ids_path datasets/paper_ids/biorxiv/2014_1_17-2022_4_30-doi-plant.txt \
        --convert_to_s2_id_path datasets/paper_ids/convert/biorxiv_2014_1_17-2022_4_30-doi-plant_to_s2_ids.json \
        --database_path /local2/hirako/s2.db \
        --output_dir "$OUTPUT_DIR" \
        --oldest_date_for_train "$OLDEST_TRAIN_YEAR" "$TEST_MONTH" \
        --test_date "$TEST_YEAR" "$TEST_MONTH" \
        --n_years_after "$N_YEARS_AFTER" \
        --mode_for_citation_counts_within_n_years_after_publication use_full

    # no_complement
    OUTPUT_DIR="datasets/ccp/biorxiv/${YEAR_RANGE}_years/use_current_citation/test_${TEST_YEAR}-${TEST_MONTH}/${N_YEARS_AFTER}_year_later_citation_no_complemented"
    echo "$OUTPUT_DIR"
    python create_ccp_dataset/create_dataset_from_db.py \
        --paper_ids_path datasets/paper_ids/biorxiv/2014_1_17-2022_4_30-doi-plant.txt \
        --convert_to_s2_id_path datasets/paper_ids/convert/biorxiv_2014_1_17-2022_4_30-doi-plant_to_s2_ids.json \
        --database_path /local2/hirako/s2.db \
        --output_dir "$OUTPUT_DIR" \
        --oldest_date_for_train "$OLDEST_TRAIN_YEAR" "$TEST_MONTH" \
        --test_date "$TEST_YEAR" "$TEST_MONTH" \
        --n_years_after "$N_YEARS_AFTER" \
        --mode_for_citation_counts_within_n_years_after_publication no_complement
done
