#!/usr/bin/env bash
# Extract the MIMIC-IV-Echo cohort for the "same pipeline, new data" check.
#
# Cohort = ED admissions (edregtime not null) with >= 1 TTE during the stay AND an ICD code
# for at least one candidate disease of the "shortness of breath / chest pain" differential.
# Candidates are kept broad on purpose; the final 4 (like the abdominal set) are chosen later
# from the counts this produces.
#
# Tables (BigQuery dataset ${dataset}, exported as Parquet to data/raw/echo_mimiciv_3_1/<table>/):
#   cohort            one row per admission: demographics, first TTE time, per-disease min seq_num
#   diagnoses         all ICD codes of those admissions (+ titles)
#   discharge_notes   discharge summaries -> Patient History (HPI / PMH / social / family)
#   radiology         radiology reports (leak check, real wording for result sentences)
#   labs_selected     BNP / NT-proBNP, troponin, D-dimer, procalcitonin (lab controls)
#   echo_measurements structured TTE/TEE/stress measurements inside the admission (+ hadm_id)
#   echo_studies      DICOM-subset study index (+ note_id where a report exists)
#   echo_notes        only if physionet-data.mimiciv_note has an echo table
#
# Needs: gcloud auth login (same Google account that holds PhysioNet BigQuery access).
# Rerun-safe: BigQuery tables use IF NOT EXISTS, and finished local tables are skipped.
# MIMIC data: never commit (data/ is gitignored). Copy to Quest with scp.
set -euo pipefail

project="congraph-mimic-ibd"
dataset="echo_extract"
bucket="congraph-mimic-ibd-export-duruoli-2026"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
local_root="${repo_root}/data/raw/echo_mimiciv_3_1"
P="${project}.${dataset}"
mkdir -p "$local_root"

run_sql() {
  bq query --project_id="$project" --use_legacy_sql=false \
    --maximum_bytes_billed=60000000000 "$@"
}

bq --project_id="$project" show "${project}:${dataset}" >/dev/null 2>&1 \
  || bq --project_id="$project" mk --dataset --location=US "${project}:${dataset}"

# ---- cohort -------------------------------------------------------------------------
run_sql <<SQL
CREATE TABLE IF NOT EXISTS \`${P}.cohort\` AS
WITH codes AS (
  SELECT * FROM UNNEST([
    STRUCT(10 AS v, 'I50' AS p, 'heart_failure' AS dz), (9, '428', 'heart_failure'),
    (10, 'I26', 'pulmonary_embolism'), (9, '4151', 'pulmonary_embolism'),
    (10, 'J12', 'pneumonia'), (10, 'J13', 'pneumonia'), (10, 'J14', 'pneumonia'),
    (10, 'J15', 'pneumonia'), (10, 'J16', 'pneumonia'), (10, 'J17', 'pneumonia'),
    (10, 'J18', 'pneumonia'),
    (9, '480', 'pneumonia'), (9, '481', 'pneumonia'), (9, '482', 'pneumonia'),
    (9, '483', 'pneumonia'), (9, '484', 'pneumonia'), (9, '485', 'pneumonia'),
    (9, '486', 'pneumonia'),
    (10, 'I33', 'endocarditis'), (9, '421', 'endocarditis'),
    (10, 'I30', 'pericardial'), (10, 'I313', 'pericardial'), (10, 'I314', 'pericardial'),
    (9, '420', 'pericardial'), (9, '4233', 'pericardial'), (9, '4239', 'pericardial'),
    (10, 'I350', 'aortic_stenosis'), (10, 'I352', 'aortic_stenosis'),
    (9, '4241', 'aortic_stenosis'),          -- ICD-9 4241 = all aortic valve disorders
    (10, 'I21', 'acute_mi'), (9, '410', 'acute_mi'),
    (10, 'J441', 'copd_exacerbation'), (9, '49121', 'copd_exacerbation')
  ])
),
matched AS (
  SELECT d.subject_id, d.hadm_id, c.dz, MIN(d.seq_num) AS seq,
         STRING_AGG(DISTINCT d.icd_code, ',') AS icd
  FROM \`physionet-data.mimiciv_3_1_hosp.diagnoses_icd\` AS d
  JOIN codes AS c
    ON d.icd_version = c.v AND STARTS_WITH(REPLACE(UPPER(d.icd_code), '.', ''), c.p)
  GROUP BY 1, 2, 3
),
per_adm AS (
  SELECT subject_id, hadm_id,
    MIN(IF(dz = 'heart_failure', seq, NULL))      AS seq_heart_failure,
    MIN(IF(dz = 'pulmonary_embolism', seq, NULL)) AS seq_pulmonary_embolism,
    MIN(IF(dz = 'pneumonia', seq, NULL))          AS seq_pneumonia,
    MIN(IF(dz = 'endocarditis', seq, NULL))       AS seq_endocarditis,
    MIN(IF(dz = 'pericardial', seq, NULL))        AS seq_pericardial,
    MIN(IF(dz = 'aortic_stenosis', seq, NULL))    AS seq_aortic_stenosis,
    MIN(IF(dz = 'acute_mi', seq, NULL))           AS seq_acute_mi,
    MIN(IF(dz = 'copd_exacerbation', seq, NULL))  AS seq_copd_exacerbation,
    STRING_AGG(dz, '+' ORDER BY seq)              AS candidate_dzs,
    STRING_AGG(IF(seq = 1, dz, NULL), '+')        AS primary_dz,
    STRING_AGG(CONCAT(dz, ':', icd), ' ' ORDER BY seq) AS matched_icd
  FROM matched GROUP BY 1, 2
),
tte AS (
  SELECT a.hadm_id, MIN(m.measurement_datetime) AS first_tte_time,
         COUNT(DISTINCT m.measurement_id) AS n_tte
  FROM \`physionet-data.mimiciv_echo.structured_measurement\` AS m
  JOIN \`physionet-data.mimiciv_3_1_hosp.admissions\` AS a
    ON m.subject_id = a.subject_id
   AND m.measurement_datetime >= a.admittime AND m.measurement_datetime < a.dischtime
  WHERE LOWER(m.test_type) = 'tte'
  GROUP BY 1
)
SELECT x.*, a.admittime, a.dischtime, a.edregtime, a.admission_type, a.admission_location,
       a.hospital_expire_flag, p.gender,
       p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS age,
       t.first_tte_time, t.n_tte
FROM per_adm AS x
JOIN tte AS t USING (hadm_id)
JOIN \`physionet-data.mimiciv_3_1_hosp.admissions\` AS a USING (subject_id, hadm_id)
JOIN \`physionet-data.mimiciv_3_1_hosp.patients\` AS p USING (subject_id)
WHERE a.edregtime IS NOT NULL;
SQL

# ---- per-admission tables -----------------------------------------------------------
run_sql <<SQL
CREATE TABLE IF NOT EXISTS \`${P}.diagnoses\` AS
SELECT d.*, dict.long_title
FROM \`physionet-data.mimiciv_3_1_hosp.diagnoses_icd\` AS d
JOIN \`${P}.cohort\` AS c USING (subject_id, hadm_id)
LEFT JOIN \`physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses\` AS dict USING (icd_code, icd_version);

CREATE TABLE IF NOT EXISTS \`${P}.discharge_notes\` AS
SELECT n.*
FROM \`physionet-data.mimiciv_note.discharge\` AS n
JOIN \`${P}.cohort\` AS c USING (subject_id, hadm_id);

CREATE TABLE IF NOT EXISTS \`${P}.radiology\` AS
SELECT n.*
FROM \`physionet-data.mimiciv_note.radiology\` AS n
JOIN \`${P}.cohort\` AS c USING (subject_id, hadm_id);

CREATE TABLE IF NOT EXISTS \`${P}.labs_selected\` AS
SELECT l.*, dict.label, dict.fluid, dict.category
FROM \`physionet-data.mimiciv_3_1_hosp.labevents\` AS l
JOIN \`${P}.cohort\` AS c USING (subject_id, hadm_id)
JOIN \`physionet-data.mimiciv_3_1_hosp.d_labitems\` AS dict USING (itemid)
WHERE REGEXP_CONTAINS(LOWER(dict.label), r'natriuretic|bnp|troponin|d-dimer|procalcitonin');

-- echo tables carry no hadm_id: attach by time inside the admission (same as the IBD script)
CREATE TABLE IF NOT EXISTS \`${P}.echo_measurements\` AS
SELECT c.hadm_id, m.*
FROM \`physionet-data.mimiciv_echo.structured_measurement\` AS m
JOIN \`${P}.cohort\` AS c
  ON m.subject_id = c.subject_id
 AND m.measurement_datetime >= c.admittime AND m.measurement_datetime < c.dischtime;

CREATE TABLE IF NOT EXISTS \`${P}.echo_studies\` AS
SELECT c.hadm_id, s.*
FROM \`physionet-data.mimiciv_echo.echo_study_list\` AS s
JOIN \`${P}.cohort\` AS c
  ON s.subject_id = c.subject_id
 AND s.study_datetime >= c.admittime AND s.study_datetime < c.dischtime;
SQL

tables=(cohort diagnoses discharge_notes radiology labs_selected echo_measurements echo_studies)

# Echo report text: PhysioNet says it "will be made available under MIMIC-IV-Note".
echo_note_table="$(bq ls --project_id="$project" --format=csv physionet-data:mimiciv_note \
  | awk -F, 'NR > 1 && tolower($1) ~ /echo/ {print $1; exit}')"
if [[ -n "$echo_note_table" ]]; then
  echo "Found echo note table: mimiciv_note.${echo_note_table}"
  run_sql <<SQL
CREATE TABLE IF NOT EXISTS \`${P}.echo_notes\` AS
SELECT n.*
FROM \`physionet-data.mimiciv_note.${echo_note_table}\` AS n
JOIN \`${P}.cohort\` AS c USING (subject_id);
SQL
  tables+=(echo_notes)
else
  echo "No echo note table in mimiciv_note (report text not released yet); skipping."
fi

# ---- export, download, validate -------------------------------------------------------
for table in "${tables[@]}"; do
  target="${local_root}/${table}"
  if [[ -e "$target" ]]; then
    echo "Skip ${table}: ${target} already exists"
    continue
  fi
  expected="$(bq query --project_id="$project" --use_legacy_sql=false --quiet --format=csv \
    "SELECT COUNT(*) AS n FROM \`${P}.${table}\`" | tail -n 1 | tr -d '\r')"
  [[ "$expected" =~ ^[0-9]+$ ]] || { echo "Bad row count for ${table}: ${expected}" >&2; exit 1; }

  export_prefix="${dataset}/${table}/$(date -u +%Y%m%dT%H%M%S)-$$"
  run_sql <<SQL
EXPORT DATA OPTIONS (
  uri = 'gs://${bucket}/${export_prefix}/part-*.parquet',
  format = 'PARQUET', compression = 'SNAPPY', overwrite = FALSE
) AS SELECT * FROM \`${P}.${table}\`;
SQL

  scratch="$(mktemp -d "${local_root}/.${table}.download.XXXXXX")"
  gcloud storage rsync --recursive "gs://${bucket}/${export_prefix}" "$scratch"
  python - "$scratch" "$expected" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq
folder, expected = Path(sys.argv[1]), int(sys.argv[2])
files = sorted(folder.glob("part-*.parquet"))
actual = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
if not files or actual != expected:
    raise SystemExit(f"Validation failed: {folder}: {actual} rows, expected {expected}")
print(f"Validated {folder.name}: {actual:,} rows in {len(files)} file(s)")
PY
  mv "$scratch" "$target"
  echo "Saved ${table}: ${expected} rows -> ${target}"
done

# ---- quick counts for choosing the final disease set -----------------------------------
python - "$local_root" <<'PY'
import sys
from pathlib import Path
import pandas as pd
c = pd.read_parquet(Path(sys.argv[1]) / "cohort")
print(f"\ncohort: {len(c):,} admissions, {c.subject_id.nunique():,} patients")
seq = [k for k in c.columns if k.startswith("seq_")]
print(pd.DataFrame({
    "any_position": c[seq].notna().sum(),
    "primary(seq=1)": (c[seq] == 1).sum(),
    "top3(seq<=3)": (c[seq] <= 3).sum(),
}).rename(index=lambda k: k[4:]).to_string())
PY
