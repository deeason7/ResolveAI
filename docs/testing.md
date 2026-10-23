# Running tests

Run commands from `backend/` with the development dependencies installed.

```sh
python -m pytest tests/ -q
python -m ruff check . ../frontend ../scripts
python -m ruff format --check . ../frontend ../scripts
```

The default suite uses a temporary SQLite file with foreign keys enabled and
in-process service fakes. PostgreSQL tests are skipped unless `--run-postgres`
is supplied. No database service is needed for the default run. SQLite and mocks
do not establish PostgreSQL migration or transaction behavior.

## Disposable PostgreSQL

The integration suite supports PostgreSQL 16; CI uses `postgres:16.3-alpine`,
matching the current Compose image. It does not run the application Compose
stack or load a deployment environment file. For an isolated local container:

```sh
docker run --name resolveai-test-postgres --rm -d \
  -e POSTGRES_DB=resolveai_test_local \
  -e POSTGRES_USER=resolveai_test \
  -e POSTGRES_PASSWORD=test-only-password \
  -p 127.0.0.1:55432:5432 postgres:16.3-alpine
docker exec resolveai-test-postgres pg_isready -U resolveai_test -d resolveai_test_local
```

Wait for `pg_isready` to report that it is accepting connections, then run:

```sh
TEST_DATABASE_URL='postgresql+asyncpg://resolveai_test:test-only-password@127.0.0.1:55432/resolveai_test_local' \
TEST_DATABASE_ALLOW_RESET=yes \
python -m pytest tests/integration/ --run-postgres -m postgres -q
```

Every integration test **drops and recreates the entire public schema** in its
configured database. Use a dedicated disposable database containing no useful
data, and run one integration process per database. Parallel processes must use
different database names; the fixture does not support sharing a target.

The guard requires all of the following before resetting data:

- Explicit `TEST_DATABASE_URL`; the application's `DATABASE_URL` is never used.
- `TEST_DATABASE_ALLOW_RESET=yes` and the `--run-postgres` option.
- The `postgresql+asyncpg` driver, a loopback host, an explicit user and no URL
  query parameters that could override connection settings.
- A database named `resolveai_test` or `resolveai_test_<lowercase_suffix>`.
- A connected `current_database()` matching that validated name.

These checks prevent common targeting mistakes, not deliberate misuse. Do not
forward a production database to a local port or rename valuable data to fit
the test naming convention. The suite never creates or drops a database, but
its schema reset destroys all objects in the disposable database's public schema.

Each `pg_engine` fixture builds schema through Alembic from empty. The
`postgres_database` fixture also exposes `upgrade(revision)` for migration
tests. The migration subprocess uses the test URL, synthetic configuration,
and a temporary working directory without a `.env` file. Its connection is
separate from the unit suite's SQLite application singleton. Use `pg_engine`
or `pg_sessionmaker` explicitly in PostgreSQL tests and mark them `postgres`.

Coverage includes an empty migration chain, a linked-data upgrade from a previous
revision, foreign-key and unique-constraint rejection, two independent database
connections racing at a synchronization barrier, and visibility before/after
commit. The migration tests check raw SQL viewer defaults and legacy classification
provenance. Classification tests persist real outage state and synchronize a human
commit during inference; recovery includes the new metadata. The uniqueness race
establishes the database backstop; it does not claim
that every API currently handles that conflict correctly.

Stop only the disposable container created above when finished:

```sh
docker stop resolveai-test-postgres
```

CI runs integration tests as a separate job in the reusable workflow.
Local success does not establish hosted CI success or branch-protection settings.

## Dependency compatibility

The application currently pins SQLModel 0.0.38 and the verified FastAPI 0.136.3 /
SlowAPI 0.1.9 pairing. A fresh CI install on 2026-10-21 selected SQLModel 0.0.48
and failed existing UTC-naive timestamp writes. SQLModel's
[0.0.45 release notes](https://sqlmodel.tiangolo.com/release-notes/#0045-2026-09-21)
describe that breaking datetime change. A timezone migration needs coordinated
model, schema and backup checks; a dependency install must not implicitly change
that contract. The newer FastAPI router representation also bypassed SlowAPI's
default-limit discovery; the refresh outage regression test caught that failure.

These focused compatibility pins do not lock the full deployment dependency graph.
A clean resolved installation and full lockfiles remain separate verification work.
