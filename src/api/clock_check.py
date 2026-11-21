"""
Server time-zone check at startup (scale plan S7, docs/plans/scale_plan.md; approved 2026-10-26).

The relative-date rule (src/verification/relative_date.py) and the old-post warning (src/verification/post_age.py)
read the server's local date. They expect Philippine time (Asia/Manila, UTC+08:00, no daylight saving; v1.3 release
notes). Hosted servers usually run on UTC, which is 8 hours behind: from 00:00 to 08:00 Philippine time such a server
still thinks it is yesterday. This module only WARNS (in the server log): it never stops the server and never changes
an answer. The real protection is TZ=Asia/Manila in the deployment.
"""
from datetime import datetime, timedelta
from typing import Optional

EXPECTED_OFFSET = timedelta(hours=8)


def _format(offset: timedelta) -> str:
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    return f"UTC{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def timezone_warning(now: Optional[datetime] = None) -> Optional[str]:
    """None when the server's local clock is UTC+08:00; otherwise the warning to log."""
    local = now if (now is not None and now.tzinfo is not None) else datetime.now().astimezone()
    offset = local.utcoffset() or timedelta(0)
    if offset == EXPECTED_OFFSET:
        return None
    return (f"Server clock is {_format(offset)}, but the date rules (relative dates, old-post warning) expect "
            f"Philippine time ({_format(EXPECTED_OFFSET)}). 'Today' may be off by up to "
            f"{abs((EXPECTED_OFFSET - offset).total_seconds()) / 3600:g} hours. Set TZ=Asia/Manila in the deployment.")
