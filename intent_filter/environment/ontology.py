"""Environment ontology: rooms, objects, roles, world variables.

Loaded from config/environment_ontology.yaml. This is the `O` (objects) and
`Pr` (properties) part of the planning problem tuple `P = <O, Pr, A, S, T, I, G, tau>`
described in the proposal's methodology (see intent_filter/environment/problem.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Room:
    name: str
    tags: frozenset[str] = field(default_factory=frozenset)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


@dataclass(frozen=True)
class EnvObject:
    name: str
    properties: frozenset[str] = field(default_factory=frozenset)
    default_room: str | None = None

    def has_property(self, prop: str) -> bool:
        return prop in self.properties


@dataclass(frozen=True)
class Ontology:
    """The full environment ontology: ground truth for what exists in the world."""

    rooms: dict[str, Room]
    objects: dict[str, EnvObject]
    roles: tuple[str, ...]
    world_variables: tuple[str, ...]

    def room(self, name: str) -> Room:
        if name not in self.rooms:
            raise KeyError(f"Unknown room: {name!r}. Known rooms: {sorted(self.rooms)}")
        return self.rooms[name]

    def obj(self, name: str) -> EnvObject:
        if name not in self.objects:
            raise KeyError(f"Unknown object: {name!r}. Known objects: {sorted(self.objects)}")
        return self.objects[name]

    def has_role(self, name: str) -> bool:
        return name in self.roles


def load_ontology(path: str | Path) -> Ontology:
    """Load and validate the environment ontology from YAML.

    Raises ValueError on malformed data (e.g. an object's default_room that
    doesn't exist) so configuration errors surface immediately at startup
    rather than as confusing failures deep in the transition function.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    rooms = {
        room_name: Room(name=room_name, tags=frozenset(room_data.get("tags") or []))
        for room_name, room_data in (raw.get("rooms") or {}).items()
    }

    objects = {}
    for obj_name, obj_data in (raw.get("objects") or {}).items():
        default_room = obj_data.get("default_room")
        if default_room is not None and default_room not in rooms:
            raise ValueError(
                f"Object {obj_name!r} has default_room {default_room!r} "
                f"which is not defined in rooms: {sorted(rooms)}"
            )
        objects[obj_name] = EnvObject(
            name=obj_name,
            properties=frozenset(obj_data.get("properties") or []),
            default_room=default_room,
        )

    roles = tuple(raw.get("roles") or [])
    world_variables = tuple(raw.get("world_variables") or [])

    return Ontology(rooms=rooms, objects=objects, roles=roles, world_variables=world_variables)
