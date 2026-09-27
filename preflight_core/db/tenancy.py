"""Tenant-scoped database access (row-level security)."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Connection, Engine, text


@contextmanager
def tenant_connection(engine: Engine, installation_id: int) -> Iterator[Connection]:
    """A transaction that can only see rows of repos in `installation_id`.

    SET LOCAL scopes both the role and the setting to this transaction, so a pooled
    connection never leaks one tenant's scope into the next checkout.
    """
    with engine.begin() as c:
        c.execute(text("SET LOCAL ROLE preflight_tenant"))
        c.execute(
            text("SELECT set_config('app.installation_id', :i, true)"),
            {"i": str(int(installation_id))},
        )
        yield c
