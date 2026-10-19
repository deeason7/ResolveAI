# Account access and provisioning

Public self-registration creates **viewers only**. It is disabled by default when
ENVIRONMENT is production, staging, or an unrecognized value. Development and test
default to enabled. PUBLIC_REGISTRATION_ENABLED explicitly overrides the default;
enabling it never grants staff privileges.

GET /api/v1/auth/capabilities exposes only the registration setting so the login
page can show the appropriate controls. The server enforces the policy even if a
client ignores the page.

| Role | Access |
|---|---|
| viewer | Authenticated read endpoints |
| analyst | Read, submit, enqueue, generate, approve and reject |
| admin | Analyst access plus bulk import |

The server loads the current account and role for every authenticated request.
Changing the database role takes effect for existing access tokens. Existing
registered analysts are not automatically demoted by this release; inspect them
and make an explicit account decision.

## Public demo

The demo must be provisioned as viewer before use. The frontend never creates it
on login failure. Login, refresh and authenticated requests reject the configured
demo identity if it has an elevated role, including previously issued tokens.

Set DEMO_EMAIL consistently in backend and frontend configuration. The frontend's
DEMO_PASSWORD must match the provisioned viewer password. Demo credentials are
public by design; they must not belong to a staff account. The standard identity
is demo@resolveai-demo.com. Changing configuration does not remediate an old
account: inventory old public identities explicitly.

## Administrative command

Run scripts/manage_users.py with the backend environment installed. It does not
load dotenv files. Set a dedicated environment variable to the intended database
URL through your normal secure local configuration; never paste credentials into
a committed file or command argument.

From the repository root, a bounded read-only inventory is:

```sh
backend/.venv/bin/python scripts/manage_users.py --database-url-env ADMIN_DATABASE_URL inventory --role analyst --limit 100
backend/.venv/bin/python scripts/manage_users.py --database-url-env ADMIN_DATABASE_URL inventory --email demo@resolveai-demo.com
```

The report contains IDs, email, role and active state, never password hashes.
A truncated inventory is explicitly marked; use exact email/ID queries to inspect
a selected account. Use --require-ssl before the subcommand for a PostgreSQL target
requiring verified TLS.

Preview demo creation:

```sh
backend/.venv/bin/python scripts/manage_users.py --database-url-env ADMIN_DATABASE_URL create --email demo@resolveai-demo.com --full-name "Demo Viewer" --role viewer
```

Add --apply to that command only after reviewing the target and preview. Creation
then prompts for a password without displaying it. For noninteractive provisioning,
--password-env names a variable holding the password; it does not accept the
password itself. The command does not read a password during dry-run or when the
same-role account already exists.

Staff accounts use the same create command with an explicitly selected analyst or
admin role and a non-demo address. Existing credentials, names and roles are never
overwritten by create. A different existing role requires a separate operation.

For a selected existing account, preview a role change:

```sh
backend/.venv/bin/python scripts/manage_users.py --database-url-env ADMIN_DATABASE_URL set-role --user-id SELECTED_UUID --expected-role analyst --role viewer
```

Replace SELECTED_UUID with the ID from inventory; add --apply after review. The
command locks and checks the expected current role, changes only that ID, and
records an audit event in the same transaction. Repeating the same completed
change has no additional effect. A mismatch requires a fresh inventory, not a
broader selector.

Changing access in a shared environment is an administrative data operation.
Take a complete backup and verify the selected account; do not infer which users
are legitimate staff from their registration date.

## Rollout and limits

Deploy backend and frontend together, set PUBLIC_REGISTRATION_ENABLED=false for a
public production demo, provision or explicitly remediate the demo viewer, and
verify read access plus denied writes. Source tests do not establish current hosted
account state.

The refresh-token rotation, password-verification byte boundary, and production
HTTP/Secure-cookie transport findings are separate outstanding fixes. New accounts
created by the administrative command reject bcrypt passwords over 72 UTF-8 bytes;
that does not claim the existing public registration/login paths are repaired yet.
