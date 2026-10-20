# Complete backup and recovery

scripts/backup_state.py writes version 2 application archives. Every registered
application table is included, including complaints and their original UUIDs.
A complaint reimport cannot recreate submissions, classifications, review state
or foreign-key relationships.

The PostgreSQL export uses one read-only REPEATABLE READ transaction. A manifest
records the schema revision, dialect, table schema, row counts and SHA-256 hashes.
All data is written before the completed manifest is published. The requested
output directory must not exist; existing archives are never overwritten.

## Export

From the repository root, with backend dependencies installed:

```sh
backend/.venv/bin/python scripts/backup_state.py --database-url-env BACKUP_DATABASE_URL --out-dir backups/complete-archive
```

Set BACKUP_DATABASE_URL explicitly using secure local configuration. The script
does not load .env files. The CLI requires a postgresql+asyncpg URL. Add
--require-ssl when the target requires verified TLS. With no --out-dir, a unique
UTC-named directory is created under backups/ relative to the working directory.

The script refuses a database without exactly one Alembic revision or with an
unexpected table inventory. Run it using application code compatible with the
source schema. A new persistent table must remain registered in app.models and
be covered by recovery tests; an unmapped database table causes a visible failure.

Archives include password hashes, email addresses and complaint text. New archive
directories are private (0700) and files use 0600. Store a protected copy outside
the database host and keep it outside Git and build contexts. Permissions and
checksums do not provide encryption or proof against deliberate tampering.

## Restore drill

1. Create a separate empty disposable database.
2. Read the manifest's alembic_revision. Run the application's Alembic migrations
   to that exact revision on the identified target, using compatible application
   code. The JSONL archive contains data and schema fingerprints, not DDL.
3. Point RESTORE_DATABASE_URL explicitly at that target, then run:

```sh
backend/.venv/bin/python scripts/backup_state.py --database-url-env RESTORE_DATABASE_URL --restore backups/complete-archive
```

4. Compare reported inserted counts with the manifest; inspect case IDs, foreign
   keys, statuses, exact draft text, classification outcomes/provenance/staleness,
   labels and audit history. Time the drill with
   the intended corpus size and record server/application/schema versions.
5. Upgrade only after the restored revision is verified. Rebuild graph/vector
   derivatives from canonical data, then verify those separately.

Restore validates every required file before inserting, checks schema/dialect
compatibility and requires all application tables empty. It takes exclusive table
locks on PostgreSQL during the restore, then inserts in foreign-key order in one
transaction. Any conflict or insert failure rolls back the entire operation.
Counts are reported as inserted only after commit.

There is no --force override, silent conflict skip, or partial-merge mode. Repeating
a completed restore against its nonempty target is rejected. Provision accounts
and start application traffic only after the restore, not during it.

## Interrupted and legacy archives

A failed export may leave an empty reserved output directory and a sibling
.partial-* staging directory. Neither has the completed manifest. Preserve these
for diagnosis, choose a new output directory for the next attempt, and never
rename a partial manifest to declare success.

Version 1 archives are rejected by the new restore path because they have no
complete integrity/schema contract and may omit complaints. Keep them intact for
a separately planned recovery. Reimporting public records or ignoring foreign-key
errors cannot repair missing original IDs and live case state.

A backup on the same host is not disaster recovery. Use supported database-level
backup as appropriate, including a compatible pg_dump client and a pg_restore
drill. Do not tar a live PostgreSQL data directory. See
[PostgreSQL logical backups](https://www.postgresql.org/docs/current/app-pgdump.html).

## Verification scope

The tests cover full roundtrip state on SQLite with foreign keys and on an empty,
Alembic-migrated PostgreSQL database. A concurrent-writer test verifies one stable
export snapshot. Unit tests reject missing/corrupt files, wrong schema/revision,
nonempty targets, legacy formats and conflicting rows without partial restoration.

The small synthetic drill establishes correctness for those fixtures. It does
not measure production recovery time, storage durability, backup scheduling or
hosted restore permissions. Follow [testing.md](testing.md) to reproduce it.


## Exact-review schema compatibility

Migration `e6b4a9d2c705` adds `review_events`, case/input versions and explicit
legacy provenance markers. Default backups include the review table through the
registered model inventory, preserving reviewer/draft relationships and hashes.
The restore drill includes a reviewed response and its immutable event.

A format-2 archive from an earlier schema must be restored with its matching
application revision into an empty database migrated to that archived revision,
then upgraded forward. The current restore refuses a mismatched table inventory
or schema revision. Do not remove review history to make an old archive fit.
Downgrading the review migration refuses to proceed once review events exist.
