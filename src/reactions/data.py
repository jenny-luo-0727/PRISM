"""Reaction data for PRISM regression: text + one 3D molecule graph per reaction component.

Reads an exported reaction table and per-molecule conformers (see
``applications/reactions/README.md``). Each reaction carries one SMILES text string, a
scalar target, and an ordered tuple of component molecules; each molecule becomes a
non-periodic radius graph built the same way as ``src.materials.data.cif_to_graph``
(edges from every atom to its ``max_neighbors`` closest atoms within ``cutoff`` Å).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.materials.data import TextTokenizer

PARTITIONS = ("train", "val", "test")


@dataclass(frozen=True)
class ReactionRecord:
    reaction_id: str
    text: str
    target: float
    molecules: tuple[str, ...]  # molecule keys, one per component, in component order


def molecule_graph(
    atomic_numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    cutoff: float = 5.0,
    max_neighbors: int = 16,
) -> dict[str, torch.Tensor]:
    """Non-periodic radius graph: each atom to its closest ``max_neighbors`` atoms within ``cutoff``."""
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be positive")
    xyz = np.asarray(positions, dtype=np.float64)
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    centers, neighbors = [], []
    for atom in range(len(xyz)):
        within = np.flatnonzero(distances[atom] <= cutoff)
        closest = within[np.argsort(distances[atom, within], kind="stable")][:max_neighbors]
        centers.extend([atom] * len(closest))
        neighbors.extend(closest.tolist())
    center = np.asarray(centers, dtype=np.int64)
    neighbor = np.asarray(neighbors, dtype=np.int64)
    return {
        "atomic_numbers": torch.tensor(list(atomic_numbers), dtype=torch.long),
        "edge_index": torch.from_numpy(np.stack((center, neighbor))),
        "edge_distance": torch.from_numpy(distances[center, neighbor].astype(np.float32)),
    }


def load_reactions(
    data_dir: Path, split: str, components: Sequence[str]
) -> tuple[dict[str, list[ReactionRecord]], dict[str, dict]]:
    """Load ``reactions.csv`` partitioned by the ``split`` column, plus ``molecules.json``.

    ``components`` names the id columns, in order, whose ``<role>:<id>`` keys index the
    molecules (e.g. ``thiol_key`` -> ``thiol:<key>``); the order sets the graph-token order.
    """
    table = pd.read_csv(data_dir / "reactions.csv", dtype=str)
    if split not in table.columns:
        raise ValueError(f"split {split!r} is not a column of reactions.csv")
    molecules = json.loads((data_dir / "molecules.json").read_text())
    roles = [column.removesuffix("_key").removesuffix("_id") for column in components]
    partitions: dict[str, list[ReactionRecord]] = {part: [] for part in PARTITIONS}
    for row in table.itertuples(index=False):
        part = getattr(row, split)
        if part not in partitions:
            continue
        keys = tuple(f"{role}:{getattr(row, column)}" for role, column in zip(roles, components, strict=True))
        missing = [key for key in keys if key not in molecules]
        if missing:
            raise KeyError(f"{row.reaction_id}: no conformer for {missing}")
        partitions[part].append(ReactionRecord(row.reaction_id, row.text, float(row.target), keys))
    if not all(partitions.values()):
        raise ValueError(f"split {split!r} has an empty partition: { {k: len(v) for k, v in partitions.items()} }")
    return partitions, molecules


def shuffled_molecule_map(records: Sequence[ReactionRecord], seed: int) -> dict[str, tuple[str, ...]]:
    """Seeded derangement over reactions: each reaction gets another reaction's whole molecule tuple.

    The text keeps the true reaction; only the graph tokens come from the wrong reaction. Used
    by the ``joint_shuffled`` control to separate "the structure helps" from "extra tokens and
    trainable parameters help".
    """
    ids = [record.reaction_id for record in records]
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(ids))
        if not np.any(order == np.arange(len(ids))):
            break
    return {ids[i]: records[order[i]].molecules for i in range(len(ids))}


class ReactionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ReactionRecord],
        tokenizer: TextTokenizer,
        graphs: dict[str, dict[str, torch.Tensor]],
        max_text_length: int,
        molecule_map: dict[str, tuple[str, ...]] | None = None,
    ):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.graphs = graphs
        self.max_text_length = max_text_length
        self.molecule_map = molecule_map

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        keys = record.molecules if self.molecule_map is None else self.molecule_map[record.reaction_id]
        return {
            "reaction_id": record.reaction_id,
            "text_ids": self.tokenizer.encode(record.text, self.max_text_length),
            "text_pad_id": getattr(self.tokenizer, "pad_token_id", 0),
            "graphs": [self.graphs[key] for key in keys],
            "target": torch.tensor(record.target, dtype=torch.float32),
        }


def collate_reactions(samples: Sequence[dict]) -> dict[str, torch.Tensor | list[str] | int]:
    """Right-pad text and pack every component graph; graph ``r * n_components + c`` is reaction r, component c."""
    n_components = len(samples[0]["graphs"])
    if any(len(sample["graphs"]) != n_components for sample in samples):
        raise ValueError("Every reaction in a batch must have the same number of components")
    pad_token_id = samples[0]["text_pad_id"]
    max_tokens = max(sample["text_ids"].numel() for sample in samples)
    text_ids = torch.full((len(samples), max_tokens), pad_token_id, dtype=torch.long)
    text_mask = torch.zeros((len(samples), max_tokens), dtype=torch.bool)
    nodes, edges, distances, graph_batch = [], [], [], []
    offset = 0
    for reaction_index, sample in enumerate(samples):
        ids = sample["text_ids"]
        text_ids[reaction_index, : ids.numel()] = ids
        text_mask[reaction_index, : ids.numel()] = True
        for component_index, graph in enumerate(sample["graphs"]):
            atoms = graph["atomic_numbers"]
            nodes.append(atoms)
            edges.append(graph["edge_index"] + offset)
            distances.append(graph["edge_distance"])
            graph_batch.append(
                torch.full((atoms.numel(),), reaction_index * n_components + component_index, dtype=torch.long)
            )
            offset += atoms.numel()
    return {
        "reaction_id": [sample["reaction_id"] for sample in samples],
        "text_ids": text_ids,
        "text_mask": text_mask,
        "atomic_numbers": torch.cat(nodes),
        "edge_index": torch.cat(edges, dim=1),
        "edge_distance": torch.cat(distances),
        "graph_batch": torch.cat(graph_batch),
        "n_components": n_components,
        "target": torch.stack([sample["target"] for sample in samples]),
    }
