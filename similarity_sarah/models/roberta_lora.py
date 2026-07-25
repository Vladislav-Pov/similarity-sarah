"""RoBERTa-base + LoRA wrapper for GLUE sequence classification.

Design constraints (so the existing SARAH / prox / FedAvg / SVRS machinery
works *unchanged*):

* The rest of the codebase feeds every model a single input tensor ``x`` and
  reads ``model(x)`` as class logits (see ``utils.compute_batch_gradient``,
  the prox solvers, ``ClassificationTask``).  We therefore accept ``x`` as a
  packed integer tensor of shape ``(B, 2, L)`` where ``x[:, 0]`` are the
  ``input_ids`` and ``x[:, 1]`` the ``attention_mask`` (see
  ``data/datasets.py::_load_glue``), unpack it here, and return ``.logits``.

* All the parameter/gradient helpers iterate ``model.parameters()`` and call
  ``torch.autograd.grad(loss, list(model.parameters()))``.  That fails on
  frozen tensors (LoRA freezes the whole backbone).  We therefore override
  ``parameters()`` / ``named_parameters()`` to expose **only the trainable
  (LoRA) tensors**, which confines the entire optimisation to the adapter
  subspace with zero changes to the core.  ``.to()`` / ``state_dict()`` use
  ``_parameters`` internally (not ``parameters()``), so the full backbone is
  still moved to the device and saved in checkpoints.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import torch
import torch.nn as nn


class RobertaLoRA(nn.Module):
    """RobertaForSequenceClassification wrapped with a PEFT LoRA adapter."""

    def __init__(
        self,
        model_path: str,
        num_classes: int,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Iterable[str] = ("query", "key", "value", "dense"),
    ) -> None:
        super().__init__()
        # Imported lazily so the transformers/peft stack is only required when
        # this model is actually selected (keeps the CIFAR path dependency-free).
        from transformers import RobertaForSequenceClassification
        from peft import LoraConfig, TaskType, get_peft_model

        # Dropout is disabled so a forward pass is DETERMINISTIC: the SARAH
        # recursion evaluates ∇f_i at w_t and w_{t-1} on the *same* minibatch
        # and differences them, which is only well-defined if two forwards on
        # identical inputs match.  (FedAvg wouldn't care, but bnfg/SVRS do.)
        base = RobertaForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_classes,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        self.model = get_peft_model(base, lora_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, L) packed [input_ids, attention_mask] as produced by the
        # GLUE loader.  Token ids/masks are integer, so cast defensively.
        input_ids = x[:, 0, :].long()
        attention_mask = x[:, 1, :].long()
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    # -- Expose ONLY trainable (LoRA) params to the optimisation machinery --
    def named_parameters(  # type: ignore[override]
        self, *args, **kwargs,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        for name, param in super().named_parameters(*args, **kwargs):
            if param.requires_grad:
                yield name, param

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:  # type: ignore[override]
        for _, param in self.named_parameters(recurse=recurse):
            yield param
