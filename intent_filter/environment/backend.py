"""Simulator backend abstraction.

The custom symbolic state machine (state.py + actions.py) is the v1
implementation, wrapped here as `SymbolicSimulatorBackend`. A future
VirtualHome/AI2-THOR backend can implement `SimulatorBackend` without any
change to the four intent-filtering pipelines, which should only ever depend
on this interface, not on the symbolic implementation directly.

Wiring an actual VirtualHome/AI2-THOR backend is explicitly out of scope
until requested - this class exists only to keep that door open cheaply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from intent_filter.environment.actions import Action, transition
from intent_filter.environment.ontology import Ontology
from intent_filter.environment.state import WorldState, derived_propositions, initial_state


class SimulatorBackend(ABC):
    """Interface a pipeline uses to execute actions and query world state."""

    @abstractmethod
    def reset(self, issuing_role: str = "owner") -> WorldState:
        """Reset the simulator to its initial state and return it."""

    @abstractmethod
    def step(self, action: Action) -> WorldState:
        """Apply `action` to the current state, returning the resulting state."""

    @abstractmethod
    def current_state(self) -> WorldState:
        """Return the current state without modifying it."""

    @abstractmethod
    def propositions(self) -> dict[str, bool]:
        """Return the derived atomic-proposition dict for the current state."""


class SymbolicSimulatorBackend(SimulatorBackend):
    """v1 backend: the lightweight custom symbolic state machine."""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology
        self._state = initial_state(ontology)

    def reset(self, issuing_role: str = "owner") -> WorldState:
        self._state = initial_state(self.ontology, issuing_role=issuing_role)
        return self._state

    def step(self, action: Action) -> WorldState:
        self._state = transition(self._state, action, self.ontology)
        return self._state

    def current_state(self) -> WorldState:
        return self._state

    def propositions(self) -> dict[str, bool]:
        return derived_propositions(self._state, self.ontology)
