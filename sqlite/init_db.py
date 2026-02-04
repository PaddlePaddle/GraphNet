import sqlite3
import re
import argparse
from pathlib import Path


def parse_timestamp(filename: str) -> int:
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{6})", filename)
    if match:
        timestamp_str = match.group(1).replace("-", "")
        return int(timestamp_str)
    return 0


def migrate(db_path: str = "sqlite/GraphNet.db", migrates_dir: str = "sqlite/migrates"):
    db_path_obj = Path(db_path)
    migrates_path = Path(migrates_dir)

    if db_path_obj.exists():
        db_path_obj.unlink()
        print(f"Deleted existing database: {db_path}")

    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    db_path_obj.touch()
    print(f"Created new database: {db_path}")

    sql_files = list(migrates_path.glob("*.sql"))
    if not sql_files:
        print(f"No migration files found in {migrates_dir}")
        return

    sql_files.sort(key=lambda f: parse_timestamp(f.name))
    print(f"Found {len(sql_files)} migration file(s)")
    print("=" * 50)
    for sql_file in sql_files:
        print(f"\nExecuting: {sql_file.name}")
        with open(sql_file, "r", encoding="utf-8") as f:
            sql_content = f.read()

        try:
            conn = sqlite3.connect(db_path)
            conn.executescript(sql_content)
            conn.commit()
            conn.close()
            print(f"  ✓ Completed: {sql_file.name}")
        except Exception as e:
            print(f"  ✗ Failed: {sql_file.name}")
            print(f"  Error: {e}")
            if Path(db_path).exists():
                Path(db_path).unlink()

    print("\n" + "=" * 50)
    print(f"Migration completed. Database: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphNet database migration tool")
    parser.add_argument(
        "--db_path",
        type=str,
        default="sqlite/GraphNet.db",
        help="Database file path (default: sqlite/GraphNet.db)",
    )
    args = parser.parse_args()
    migrate(args.db_path)
