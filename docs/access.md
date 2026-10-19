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

Refresh tokens now contain unique IDs and a fingerprint of the current credential.
Rotation consumes one token atomically in Redis and retains its spent marker only
until its original expiry. Concurrent refresh attempts have one winner; separate
logins remain independent. Logout revokes the presented refresh token even after
access expiry. It does not revoke every token in a session family.

This release deliberately rejects old refresh tokens without the new claims.
Users must sign in again when their access token expires. Password reset also
invalidates prior refresh credentials. Already-issued access tokens remain valid
until their configured expiry (30 minutes by default), unless the account is
disabled. Do not represent logout/reset as immediate global access revocation.

Redis must retain spent-token markers until expiry. Configure persistence and a
no-eviction policy with monitored capacity. Production Compose enables AOF with
appendfsync=always and noeviction; this costs write latency and still needs a
persistent volume and capacity monitoring. Verify hosted Redis retention separately.
See [Redis persistence trade-offs](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/).
Redis data loss can permit replay of
still-valid tokens; after loss, rotate the refresh signing secret and require new
logins. A volume alone does not establish that every acknowledged write is durable.
Network/storage failures return 503 without issuing replacement credentials.

Production frontend requests require verified HTTPS. The Compose frontend uses
Caddy's private port 8443 with the public domain certificate; 8443 is not published.
Hosted frontends must set ENVIRONMENT=production and an HTTPS API_URL. See
[deployment](deployment.md) for certificate and routing verification.

## Password recovery

Registration and verification enforce the bcrypt limit of 72 UTF-8 bytes, not 72
characters. Over-limit login attempts are rejected before hashing or comparison;
validation responses omit submitted values. A legacy hash cannot reveal whether
its original input was longer. Do not guess affected accounts or suggest that
users sign in with a truncated prefix. Direct them to an administrator for reset.

After verifying the person's identity and selecting the account from inventory,
preview a reset:

```sh
backend/.venv/bin/python scripts/manage_users.py --database-url-env ADMIN_DATABASE_URL reset-password --user-id SELECTED_UUID --expected-email selected@example.com --reason "Verified account recovery"
```

The default dry run neither reads a new password nor changes the account. Add
--apply after reviewing the exact ID/email match; a hidden prompt requests the new
password. Noninteractive use can select --password-env NAME. The command changes
only that account, preserves its role, and records the reason in the same
transaction. Never put passwords or personal recovery evidence in the reason.
A synthetic test rehearses recovery from a legacy truncated-input hash and checks
that other accounts stay unchanged; no real accounts have been reset.

Credential routes retain a peer-address ceiling and add a shared five-attempt
account allowance per fixed minute window. The account key is HMAC-derived, and
unknown accounts consume the same allowance. A minute boundary permits a bounded
burst; this is not a complete abuse-prevention system. The peer ceiling can still
be shared by Streamlit users, and SlowAPI's Redis storage remains synchronous.
