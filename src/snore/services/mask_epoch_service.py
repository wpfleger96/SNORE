"""Mask epoch service: contiguous device-reported mask-type periods."""

from snore.services._base import ProfileScopedService
from snore.services.schemas import MaskEpochResponse

__all__ = ["MaskEpochService"]

# Maps device-reported mask_type values to the normalized mask_log style vocabulary.
# Device values not present here map to style=None.
# Keep in sync with: api/schemas.py MaskStyle, DB CHECKs (models.py, migrations 008/009), ui/src/utils/maskOptions.ts.
_MASK_TYPE_TO_STYLE: dict[str, str] = {
    "Pillows": "pillows",
    "Nasal": "nasal",
    "Full Face": "full_face",
}


class MaskEpochService(ProfileScopedService):
    """Service for querying contiguous device-reported mask-type epochs."""

    async def list_epochs(self) -> list[MaskEpochResponse]:
        """Return contiguous device mask-type epochs in chronological order.

        Each epoch is a run of consecutive nights where the device reported the
        same mask_type.  The style field normalizes the device value to the
        mask_log vocabulary; it is None for unrecognized device values (e.g.
        "Unknown").
        """
        from snore.analysis.rx_tracker import RxTracker  # noqa: PLC0415

        periods = await RxTracker(self.profile_id).get_history(
            self.db_session, keys=("mask_type",)
        )
        return [
            MaskEpochResponse(
                mask_type=p.settings["mask_type"],
                style=_MASK_TYPE_TO_STYLE.get(p.settings["mask_type"]),
                start_date=p.start_date,
                end_date=p.end_date,
                days_count=p.days_count,
                device_id=p.device_id,
                device_name=p.device_name,
            )
            for p in periods
            if "mask_type" in p.settings
        ]
