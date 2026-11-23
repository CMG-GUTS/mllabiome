from pathlib import Path
import csv
import io
import math
import os
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "campp_submission.zip"
RUNS = {
    "crc": "CAMPP-CRC-LODO-CM-official",
    "ghs": "CAMPP-GHS-NCV-53-repeated2",
    "ibd": "CAMPP-IBD-LODO-CM-official",
    "scz": "CAMPP-SCZ-NCV-CM-official",
    "dm90": "CAMPP-DM90-LODO-CM-official",
    "dm7": "CAMPP-DM7-LODO-CM-official",
}


def convert(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not {"sample_id", "mpma_b_probability_1"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns: {path}")

        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["sample_id", "prediction"])
        seen = set()
        count = 0

        for row in reader:
            if None in row:
                raise ValueError(f"Malformed row in {path}")
            sample_id = row["sample_id"]
            prediction = row["mpma_b_probability_1"]
            if not sample_id or not sample_id.strip() or sample_id in seen:
                raise ValueError(f"Invalid or duplicate sample_id in {path}: {sample_id!r}")
            try:
                value = float(prediction)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prediction in {path}: {prediction!r}") from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Prediction outside [0, 1] in {path}: {prediction!r}")
            seen.add(sample_id)
            writer.writerow([sample_id, prediction])
            count += 1

        if count == 0:
            raise ValueError(f"No predictions in {path}")
        return buffer.getvalue().encode("utf-8"), count


def main():
    contents = {}
    for task, run in RUNS.items():
        path = ROOT / "runs" / run / "inference" / "predictions.tsv"
        data, count = convert(path)
        contents[f"{task}.csv"] = data
        print(f"{task}: {count} predictions")

    with tempfile.NamedTemporaryFile(dir=ROOT, suffix=".zip", delete=False) as tmp:
        temp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for filename, data in contents.items():
                archive.writestr(filename, data)
        with zipfile.ZipFile(temp_path) as archive:
            if archive.namelist() != list(contents) or archive.testzip() is not None:
                raise ValueError("Submission ZIP validation failed")
        os.replace(temp_path, OUTPUT)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()
