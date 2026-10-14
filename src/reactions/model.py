"""PRISM regression over reaction text plus per-component molecule-graph tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.materials.prism_model import PRISMMaterialRegressor

MODES = ("text", "graph", "joint", "joint_shuffled")


class PRISMReactionRegressor(PRISMMaterialRegressor):
    """Scalar head over PRISM-fused text and one block of graph tokens per reaction component.

    Reuses the materials regressor's graph encoder, projector, and head. Each component
    molecule is encoded separately by the shared graph encoder into ``graph_tokens`` tokens; a
    learned per-component embedding is added so the backbone can tell the components apart, and
    the blocks are concatenated in component order ahead of the text tokens.
    ``joint_shuffled`` runs exactly like ``joint``; the shuffling happens in the data.
    """

    def __init__(self, prism: nn.Module, n_components: int, **kwargs):
        super().__init__(prism, **kwargs)
        self.n_components = n_components
        hidden_dim = self.prism.encoders["graph"].output_dim
        self.component_embedding = nn.Embedding(n_components, hidden_dim)
        nn.init.normal_(self.component_embedding.weight, std=0.02)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        mode: str = "joint",
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if batch["n_components"] != self.n_components:
            raise ValueError(f"batch has {batch['n_components']} components, model expects {self.n_components}")

        batch_size = batch["text_ids"].shape[0]
        device_type = batch["text_ids"].device.type
        # The graph encoder, token assembly, and head run in fp32 even inside the caller's
        # autocast (the encoder's index_add_ rejects mixed dtypes); only the LLM pass autocasts.
        with torch.autocast(device_type=device_type, enabled=False):
            inputs_embeds, attention_mask = self._assemble(batch, mode, batch_size)
        outputs = self._decoder()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state
        # Text is right-padded and graph tokens are never padding, so the last valid
        # position is (number of valid tokens - 1).
        final_index = attention_mask.sum(dim=1).clamp_min(1) - 1
        pooled = hidden[torch.arange(batch_size, device=hidden.device), final_index]
        with torch.autocast(device_type=device_type, enabled=False):
            predictions = self.regression_head(pooled.float()).squeeze(-1)
            loss = None if targets is None else F.smooth_l1_loss(predictions, targets.float())
        return predictions, loss

    def _assemble(
        self, batch: dict[str, torch.Tensor], mode: str, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Graph tokens (per component, in order) ahead of text tokens, cast to the backbone dtype."""
        embeddings = []
        masks = []
        if mode != "text":
            graph = self.prism.encoders["graph"](
                {
                    "atomic_numbers": batch["atomic_numbers"],
                    "edge_index": batch["edge_index"],
                    "edge_distance": batch["edge_distance"],
                    "graph_batch": batch["graph_batch"],
                    "batch_size": batch_size * self.n_components,
                }
            )  # (batch * components, tokens, hidden)
            tokens = graph.shape[1]
            graph = graph.view(batch_size, self.n_components, tokens, -1)
            graph = graph + self.component_embedding.weight[None, :, None, :]
            graph = self.prism.projectors["graph"](graph.view(batch_size, self.n_components * tokens, -1))
            embeddings.append(graph)
            masks.append(torch.ones(graph.shape[:2], dtype=torch.long, device=graph.device))

        if mode != "graph":
            text = self.prism.backbone.get_input_embeddings()(batch["text_ids"])
            embeddings.append(text)
            masks.append(batch["text_mask"].to(dtype=torch.long, device=text.device))

        backbone = self.prism.backbone
        target_device = next(backbone.parameters()).device
        target_dtype = backbone.get_input_embeddings().weight.dtype
        embeddings = [value.to(device=target_device, dtype=target_dtype) for value in embeddings]
        attention_mask = torch.cat([value.to(device=target_device) for value in masks], dim=1)
        return torch.cat(embeddings, dim=1), attention_mask
