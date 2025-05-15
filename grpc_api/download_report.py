import os
import shutil
import argparse
import zipfile
from glob import glob


def find_report_dir(job_id: str, reports_root: str = "reports") -> str:
    """
    Finds the latest report directory that matches the job ID prefix.
    """
    for root, dirs, files in os.walk(reports_root):
        for d in dirs:
            if job_id in d or d.startswith(job_id):
                return os.path.join(root, d)
    return None


def zip_report_folder(folder_path: str, output_path: str):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                arc_path = os.path.relpath(abs_path, start=folder_path)
                zipf.write(abs_path, arc_path)


def main(job_id, output_file):
    report_dir = find_report_dir(job_id)
    if not report_dir or not os.path.exists(report_dir):
        print(f"No report found for job ID {job_id}")
        return

    zip_report_folder(report_dir, output_file)
    print(f"Packaged report to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and zip zk proof report for a given job ID.")
    parser.add_argument("--job_id", required=True, help="Job ID (or job name)")
    parser.add_argument("--output", required=False, default=None, help="Output zip file path")

    args = parser.parse_args()
    zip_name = args.output or f"{args.job_id}_zk_report.zip"
    main(args.job_id, zip_name)
