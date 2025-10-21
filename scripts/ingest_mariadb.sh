#!/bin/bash
# Ingest MariaDB dumps into the database
# Usage: ./scripts/ingest_mariadb.sh [dump_file.sql.zst]

set -e

DUMP_FILE=${1:-"data/maria/annas_archive_mariadb.sql.zst"}
DB_HOST=${MYSQL_HOST:-mariadb}
DB_NAME=${MYSQL_DATABASE:-annas_archive}
DB_USER=${MYSQL_USER:-annas_user}
DB_PASS=${MYSQL_PASSWORD:-annas_pass}

echo "Ingesting MariaDB dump: $DUMP_FILE"
echo "Database: $DB_HOST/$DB_NAME"

if [[ ! -f "$DUMP_FILE" ]]; then
    echo "Error: Dump file not found: $DUMP_FILE"
    echo "Please place your MariaDB dump in data/maria/ and run:"
    echo "  ./scripts/ingest_mariadb.sh data/maria/your_dump.sql.zst"
    exit 1
fi

# Check if file is compressed
if [[ "$DUMP_FILE" == *.zst ]]; then
    echo "Decompressing and loading zstd file..."
    zstd -d "$DUMP_FILE" -c | mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME"
else
    echo "Loading SQL file..."
    mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" < "$DUMP_FILE"
fi

echo "MariaDB ingestion completed!"
echo "You can now run title matching with:"
echo "  python src/book_download/title_matcher_cli.py --backend mariadb --in data/your_titles.csv"
