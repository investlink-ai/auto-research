"""Feast entities. See ``docs/DATA_MODEL.md`` §2."""

from __future__ import annotations

from feast import Entity, ValueType

entity_id = Entity(
    name="entity_id",
    value_type=ValueType.STRING,
    description="Tradeable ticker symbol; universe-managed",
    join_keys=["entity_id"],
)
