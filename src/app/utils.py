from datetime import datetime, timezone
from typing import Tuple


def ymdh_from_iso(dt_iso: str) -> Tuple[str, str, datetime]:
    """
    Convert ISO timestamp (e.g. '2025-09-15T09:23:45Z') into:
      - ymd (YYYYMMDD)
      - hour (HH)
      - normalized datetime object (UTC, hour precision)

    Returns:
        (ymd, hour, dt_obj)
    """
    # Ensure timezone aware
    dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
    dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)

    return dt.strftime("%Y%m%d"), dt.strftime("%H"), dt
