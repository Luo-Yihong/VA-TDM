
import copy
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from typing import Any, Dict, Optional


def create_joint_output_heads(transformer):
    """Independent [score, moment] heads initialized from the pretrained head."""
    return torch.nn.ModuleList([
        torch.nn.ModuleDict({
            "norm_out": copy.deepcopy(transformer.norm_out),
            "proj_out": copy.deepcopy(transformer.proj_out),
        }) for _ in range(2)
    ]).requires_grad_(True)


def create_zero_init_time_text_embed(transformer):
    """
    Create a copy of transformer.time_text_embed (CombinedTimestepTextProjEmbeddings)
    where ONLY the final Linear of each sub-branch (timestep_embedder.linear_2
    and text_embedder.linear_2) is zero-initialized.

    Why not zero ALL parameters?
        time_text_embed内部是 (Linear -> SiLU -> Linear)。如果所有W/b都置0:
            forward: W2*SiLU(W1*x+b1)+b2 = 0   ✓
            backward 对 W2 的梯度 = grad ⊗ SiLU(0) = 0    ❌
            backward 通过 W2 回传到 W1 的梯度 = W2^T·grad = 0  ❌
        结果：除了 linear_2.bias 外所有参数都拿不到梯度，
        time_text_embed_2 学完只是一个与 target_timestep 无关的常数偏置，
        target_timestep 对预测完全没影响。

    Only zeroing linear_2 (weight + bias):
        forward 仍为 0（init时 temb_2 = 0，旧行为保留）
        ∂L/∂W2 = grad ⊗ SiLU(linear_1(x)) ≠ 0    ✓
        梯度能通过 W2 回传到 linear_1，整个模块都能学习对
        target_timestep 的依赖。
    """
    base = transformer
    for attr_name in ("base_model", "model"):
        if hasattr(base, attr_name):
            base = getattr(base, attr_name)

    original = base.time_text_embed
    time_text_embed_2 = copy.deepcopy(original)

    # Zero the LAST linear of each Linear->SiLU->Linear branch only.
    if hasattr(time_text_embed_2, "timestep_embedder") and hasattr(time_text_embed_2.timestep_embedder, "linear_2"):
        time_text_embed_2.timestep_embedder.linear_2.weight.data.zero_()
        if time_text_embed_2.timestep_embedder.linear_2.bias is not None:
            time_text_embed_2.timestep_embedder.linear_2.bias.data.zero_()
    if hasattr(time_text_embed_2, "text_embedder") and hasattr(time_text_embed_2.text_embedder, "linear_2"):
        time_text_embed_2.text_embedder.linear_2.weight.data.zero_()
        if time_text_embed_2.text_embedder.linear_2.bias is not None:
            time_text_embed_2.text_embedder.linear_2.bias.data.zero_()

    return time_text_embed_2


def sd3_transformer_forward_joint_v2(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    target_timestep: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    return_both=None,
    use_joint_heads=False,
    joint_head_index=0,
):
    """
    SD3Transformer forward with an optional second timestep for the
    2-timestep fake-score network.

    - target_timestep is not None (and the model has `time_text_embed_2`):
      temb = time_text_embed(timestep) + time_text_embed_2(target_timestep).
      time_text_embed_2 is zero-initialized, so this starts as the identity.
    - use_joint_heads: run the trunk once, then apply the independent
      `joint_heads` with the same features and time embedding. return_both
      returns [score, moment]; otherwise joint_head_index selects one head
      (0: score, 1: moment).
    - Otherwise: the original single-timestep behavior.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)
    temb = self.time_text_embed(timestep, pooled_projections)
    if target_timestep is not None and hasattr(self, "time_text_embed_2"):
        temb = temb + self.time_text_embed_2(target_timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    def _unpatchify(hs):
        patch_size = self.config.patch_size
        h = height // patch_size
        w = width // patch_size
        hs = hs.reshape(
            shape=(hs.shape[0], h, w, patch_size, patch_size, self.out_channels)
        )
        hs = torch.einsum("nhwpqc->nchpwq", hs)
        hs = hs.reshape(
            shape=(hs.shape[0], self.out_channels, h * patch_size, w * patch_size)
        )
        return hs

    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, temb, joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

    if use_joint_heads:
        outputs = []
        heads = self.joint_heads if return_both else [self.joint_heads[joint_head_index]]
        for head in heads:
            output = _unpatchify(head["proj_out"](head["norm_out"](hidden_states, temb)))
            outputs.append(Transformer2DModelOutput(sample=output) if return_dict else (output,))
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        return tuple(outputs) if return_both else outputs[0]

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)
    output = _unpatchify(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
