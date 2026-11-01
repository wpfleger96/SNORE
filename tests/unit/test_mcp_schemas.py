"""Unit tests for MCP Pydantic schemas and schema map."""

from __future__ import annotations

import pytest

from pydantic import ValidationError

from snore.mcp.schemas import (
    SCHEMA_MODEL_MAP,
    ComplianceFields,
    DeviceInfo,
    EventRow,
    EventsResponse,
    ExperimentalNightlyRow,
    NightlyRow,
    model_to_schema,
)
from snore.provenance import Provenance, field_provenance


class TestSchemaModelMap:
    def test_all_expected_keys_present(self) -> None:
        expected = {
            "device_capabilities",
            "device_info",
            "data_overview",
            "settings_epoch",
            "settings_timeline",
            "settings_change_entry",
            "settings_changes",
            "nightly_row",
            "compliance_fields",
            "nightly_summary",
            "experimental_nightly_row",
            "experimental_nightly_metrics",
            "event_context",
            "event_row",
            "events_response",
            "experimental_event_context_row",
            "experimental_event_context_response",
            "capability_entry",
        }
        assert expected <= set(SCHEMA_MODEL_MAP.keys())

    def test_stage2_schema_keys_present(self) -> None:
        """All Stage-2 schema keys are registered in SCHEMA_MODEL_MAP."""
        stage2_keys = {
            "breath_table_query",
            "breath_table_row",
            "breath_table_bin",
            "breath_table_response",
            "window_row",
            "session_coverage_entry",
            "find_windows_response",
            "epoch_spec",
            "epoch_distribution",
            "epoch_stats",
            "epoch_rx_violation",
            "experimental_compare_epochs_response",
            "device_epoch_distribution",
            "device_epoch_stats",
            "compare_epochs_response",
        }
        assert stage2_keys <= set(SCHEMA_MODEL_MAP.keys())

    def test_model_to_schema_returns_dict_with_properties(self) -> None:
        schema = model_to_schema(DeviceInfo)
        assert "properties" in schema
        assert "id" in schema["properties"]

    def test_all_models_produce_valid_json_schema(self) -> None:
        for name, model in SCHEMA_MODEL_MAP.items():
            schema = model_to_schema(model)
            assert "type" in schema or "properties" in schema or "$defs" in schema, (
                f"Schema for {name!r} lacks type/properties/$defs"
            )


class TestNightlyRowNullReasons:
    def test_leak_above_24_pct_nullable_with_reason(self) -> None:
        from datetime import date

        row = NightlyRow(
            date=date(2024, 1, 1),
            leak_above_24_pct=None,
            leak_above_24_pct_reason="channel_absent",
        )
        assert row.leak_above_24_pct is None
        assert row.leak_above_24_pct_reason == "channel_absent"

    def test_device_summary_values_nullable_with_reason(self) -> None:
        from datetime import date

        row = NightlyRow(
            date=date(2024, 1, 1),
            ti_median_s=None,
            ti_median_reason="channel_absent",
            ie_ratio_median=None,
            ie_ratio_reason="channel_absent",
        )
        assert row.ti_median_s is None
        assert row.ie_ratio_median is None

    def test_all_optional_fields_default_to_none(self) -> None:
        from datetime import date

        row = NightlyRow(date=date(2024, 1, 1))
        assert row.ahi is None
        assert row.reras is None
        assert row.leak_above_24_pct is None
        assert row.leak_median_lpm is None
        assert row.spo2_mean_pct is None


class TestExperimentalNightlyRow:
    def test_nullable_with_reason(self) -> None:
        from datetime import date

        row = ExperimentalNightlyRow(
            date=date(2024, 1, 1),
            rera_index=None,
            rera_index_reason="not_available",
            fl_class_ge4_pct=None,
            fl_class_ge4_pct_reason="not_available",
        )
        assert row.rera_index is None
        assert row.rera_index_reason == "not_available"
        assert row.fl_class_ge4_pct_reason == "not_available"

    def test_every_metric_field_is_tagged_experimental(self) -> None:
        for name in ExperimentalNightlyRow.model_fields:
            if name.endswith("_reason") or name in {
                "date",
                "device_id",
                "rera_proxy_version",
            }:
                continue
            assert field_provenance(ExperimentalNightlyRow, name) == (
                Provenance.EXPERIMENTAL
            ), name


class TestComplianceFields:
    def test_compliance_fields_round_trip(self) -> None:
        cf = ComplianceFields(
            threshold_hours=4.0,
            days_compliant=25,
            days_total=30,
            compliance_pct=83.3,
        )
        d = cf.model_dump()
        assert d["days_compliant"] == 25
        assert d["compliance_pct"] == 83.3

    def test_compliance_fields_serializes_to_json(self) -> None:
        import json

        cf = ComplianceFields(
            threshold_hours=4.0,
            days_compliant=25,
            days_total=30,
            compliance_pct=83.3,
        )
        payload = json.loads(cf.model_dump_json())
        assert payload["threshold_hours"] == 4.0


class TestEventRowSchema:
    def test_session_id_and_session_start_wall_clock_are_required(self) -> None:
        """EventRow.session_id and session_start_wall_clock are required fields (no default)."""
        # Constructing without session_id should raise ValidationError
        with pytest.raises(ValidationError):
            EventRow(
                event_type="OA",
                start_time_wall_clock="2024-01-01T22:00:00",
                offset_seconds=0.0,
                # session_id and session_start_wall_clock omitted
            )

    def test_event_row_roundtrip_with_session_anchors(self) -> None:
        """EventRow serializes session_id and session_start_wall_clock correctly."""
        import json

        row = EventRow(
            session_id=42,
            session_start_wall_clock="2024-01-01T21:00:00",
            event_type="CA",
            start_time_wall_clock="2024-01-01T22:30:00",
            offset_seconds=5400.0,
        )
        payload = json.loads(row.model_dump_json())
        assert payload["session_id"] == 42
        assert payload["session_start_wall_clock"] == "2024-01-01T21:00:00"
        assert payload["event_type"] == "CA"

    def test_event_row_schema_contains_session_anchor_fields(self) -> None:
        """JSON schema for EventRow includes session_id and session_start_wall_clock."""
        schema = model_to_schema(EventRow)
        props = schema.get("properties", {})
        assert "session_id" in props
        assert "session_start_wall_clock" in props


class TestEventsResponseSchema:
    def test_response_level_anchors_are_nullable(self) -> None:
        """EventsResponse.session_id and session_start_wall_clock default to None."""
        resp = EventsResponse(date="2024-01-01", events=[], total_events=0)
        assert resp.session_id is None
        assert resp.session_start_wall_clock is None

    def test_response_anchors_populated_for_single_session(self) -> None:
        """EventsResponse accepts non-null anchors when all events share a session."""
        resp = EventsResponse(
            date="2024-01-01",
            session_id=7,
            session_start_wall_clock="2024-01-01T21:00:00",
            events=[],
            total_events=0,
        )
        assert resp.session_id == 7
        assert resp.session_start_wall_clock == "2024-01-01T21:00:00"

    def test_events_response_schema_nullable_anchors(self) -> None:
        """JSON schema for EventsResponse marks session anchors as nullable."""
        schema = model_to_schema(EventsResponse)
        props = schema.get("properties", {})
        assert "session_id" in props
        assert "session_start_wall_clock" in props


class TestDeviceCapabilitiesOnSchemas:
    def test_nightly_summary_response_has_device_capabilities_field(self) -> None:
        """NightlySummaryResponse includes an optional device_capabilities field."""
        from snore.mcp.schemas import NightlySummaryResponse  # noqa: PLC0415

        schema = model_to_schema(NightlySummaryResponse)
        props = schema.get("properties", {})
        assert "device_capabilities" in props

    def test_settings_timeline_has_device_capabilities_by_device_field(self) -> None:
        """SettingsTimelineResponse has top-level device_capabilities_by_device dict."""

        from snore.mcp.schemas import SettingsTimelineResponse  # noqa: PLC0415

        schema = model_to_schema(SettingsTimelineResponse)
        props = schema.get("properties", {})
        assert "device_capabilities_by_device" in props
        assert "device_capabilities" not in model_to_schema(
            __import__("snore.mcp.schemas", fromlist=["SettingsEpoch"]).SettingsEpoch
        ).get("properties", {})

    def test_events_response_has_device_capabilities_field(self) -> None:
        """EventsResponse includes an optional device_capabilities field."""
        schema = model_to_schema(EventsResponse)
        props = schema.get("properties", {})
        assert "device_capabilities" in props

    def test_device_capabilities_in_schema_model_map(self) -> None:
        """device_capabilities is exposed as a named schema in SCHEMA_MODEL_MAP."""
        assert "device_capabilities" in SCHEMA_MODEL_MAP

    def test_device_capabilities_schema_has_required_fields(self) -> None:
        """DeviceCapabilities schema includes expected boolean capability flags."""
        from snore.mcp.schemas import DeviceCapabilities  # noqa: PLC0415

        schema = model_to_schema(DeviceCapabilities)
        props = schema.get("properties", {})
        assert "has_flow_waveform" in props
        assert "has_pressure_waveform" in props
        assert "has_events" in props
        assert "has_analysis" in props


class TestSettingsEpochDeviceId:
    def test_device_id_accepts_none(self) -> None:
        """SettingsEpoch.device_id is nullable (was changed from int to int | None)."""
        from datetime import date

        from snore.mcp.schemas import SettingsEpoch

        epoch = SettingsEpoch(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            nights=90,
            settings={"mode": "CPAP"},
            device_id=None,
        )
        assert epoch.device_id is None

    def test_device_id_accepts_integer(self) -> None:
        from datetime import date

        from snore.mcp.schemas import SettingsEpoch

        epoch = SettingsEpoch(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            nights=90,
            settings={"mode": "CPAP"},
            device_id=7,
        )
        assert epoch.device_id == 7


class TestEventsResponseTruncated:
    def test_truncated_defaults_to_false(self) -> None:
        """EventsResponse.truncated defaults to False when not specified."""
        resp = EventsResponse(date="2024-01-01", events=[], total_events=0)
        assert resp.truncated is False

    def test_truncated_can_be_set_true(self) -> None:
        resp = EventsResponse(
            date="2024-01-01", events=[], total_events=10, truncated=True
        )
        assert resp.truncated is True

    def test_truncated_serializes_in_json_dump(self) -> None:
        import json

        resp = EventsResponse(date="2024-01-01", events=[], total_events=0)
        payload = json.loads(resp.model_dump_json())
        assert payload["truncated"] is False
