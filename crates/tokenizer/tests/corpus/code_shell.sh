#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/var/backups/$(hostname -s)}"
KEEP=7
mkdir -p "$BACKUP_DIR"

for db in $(psql -Atc "SELECT datname FROM pg_database WHERE NOT datistemplate"); do
    out="$BACKUP_DIR/${db}_$(date +%Y%m%d-%H%M%S).sql.zst"
    echo "dumping $db -> $out" >&2
    pg_dump --no-owner "$db" | zstd -T0 -19 > "$out" || { echo "failed: $db" >&2; continue; }
done

# delete all but the newest $KEEP files per database
find "$BACKUP_DIR" -name '*.sql.zst' -printf '%T@ %p\n' | sort -rn | awk -v keep="$KEEP" 'NR > keep { print $2 }' | xargs -r rm -v --
[[ -t 1 ]] && printf '\033[1;32mdone\033[0m\n' || echo done
