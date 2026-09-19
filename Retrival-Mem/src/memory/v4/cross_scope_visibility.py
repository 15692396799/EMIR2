from __future__ import annotations

from collections.abc import Set


def is_cross_scope_visible(
    anchor_scope_participants: Set[str],
    anchor_evidence_participants: Set[str],
    neighbor_scope_participants: Set[str],
    neighbor_evidence_participants: Set[str],
) -> bool:
    """Return whether two nodes may participate in cross-scope navigation.

    Visibility is bidirectional: each scope must contain every participant that
    supplied evidence for the node in the other scope. Missing evidence is
    treated as no consent and therefore fails closed.
    """
    return bool(
        anchor_evidence_participants
        and neighbor_evidence_participants
        and anchor_evidence_participants <= neighbor_scope_participants
        and neighbor_evidence_participants <= anchor_scope_participants
    )
