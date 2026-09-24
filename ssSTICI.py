#!/usr/bin/env python3
"""ssSTICI: sample-specific STICI for sparse scRNA-seq cell-variant matrices.

This implementation targets TensorFlow 2.14 and is adapted from the STICI
Split-Transformer with Integrated Convolutions architecture.  It trains a
separate model de novo for each sparse cell-variant matrix represented as FASTA.

Key implementation notes
------------------------
* Input alphabet: A, T, G, C, ? (missing/masked).
* Output alphabet: A, T, G, C (four-way softmax; no missing output class).
* Training corruption: a fixed fraction of observed bases is replaced by ?.
* Training loss: categorical cross-entropy + KL divergence on originally
  observed target positions.  No MaCH-Rsq loss is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import random
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import constraints, initializers, layers, regularizers
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


MODEL_NAME = "ssSTICI"
CODE_VERSION = "1.0-clean"
ALPHABET = "ATGC?"
MISSING_SYMBOL = "?"
MISSING_VALUE = ALPHABET.index(MISSING_SYMBOL)
SEQUENCE_DEPTH = len(ALPHABET)

print(f"{MODEL_NAME} {CODE_VERSION}")
print(f"TensorFlow version {tf.__version__}")


class bcolors:
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def pprint(text: str) -> None:
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")


def warn(text: str) -> None:
    print(f"{bcolors.WARNING}WARNING: {text}{bcolors.ENDC}", file=sys.stderr)


def str_to_bool(value: Union[str, bool, int]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"Invalid boolean integer: {value}")
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(
        f"Invalid boolean value: {value!r}. Use true/false, yes/no, or 1/0."
    )


def seed_everything(seed: int, deterministic_ops: bool = False) -> None:
    """Seed Python, NumPy, and TensorFlow for reproducible ssSTICI runs."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
            pprint("TensorFlow deterministic operations enabled.")
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            raise RuntimeError(
                "--deterministic-ops was requested, but TensorFlow could not enable "
                "deterministic operations."
            ) from exc


def sha256_file(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as fin:
        for chunk in iter(lambda: fin.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_versions() -> Dict[str, str]:
    build_info = tf.sysconfig.get_build_info()
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": getattr(keras, "__version__", "bundled-with-tensorflow"),
        "tensorflow_addons": tfa.__version__,
        "numpy": np.__version__,
        "tensorflow_cuda_build": str(build_info.get("cuda_version", "unknown")),
        "tensorflow_cudnn_build": str(build_info.get("cudnn_version", "unknown")),
    }


def _strip_legacy_keys(config: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    clean = dict(config)
    for key in keys:
        clean.pop(key, None)
    return clean


# -----------------------------------------------------------------------------
# Custom layers
# -----------------------------------------------------------------------------

@keras.utils.register_keras_serializable(package="ssSTICI")
class CrossAttentionLayer(layers.Layer):
    """Cross-attention block used inside each ssSTICI internal chunk."""

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        start_offset: int = 0,
        end_offset: int = 0,
        activation: Union[str, Dict[str, Any]] = "gelu",
        n_heads: int = 8,
        dropout_rate: float = 0.0,  # legacy compatibility; not applied in original block
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.local_dim = int(local_dim)
        self.global_dim = int(global_dim)
        self.start_offset = int(start_offset)
        self.end_offset = int(end_offset)
        self.n_heads = int(n_heads)
        self.dropout_rate = float(dropout_rate)
        self.activation = keras.activations.get(activation)

        self.layer_norm00 = layers.LayerNormalization()
        self.layer_norm01 = layers.LayerNormalization()
        self.layer_norm1 = layers.LayerNormalization()
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.local_dim // 2, activation=self.activation),
                layers.Dense(self.local_dim, activation=self.activation),
            ],
            name="cross_attention_ffn",
        )
        self.add0 = layers.Add()
        self.add1 = layers.Add()
        # key_dim intentionally matches the historical ssSTICI/STICI implementation.
        self.attention = layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.local_dim,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "local_dim": self.local_dim,
                "global_dim": self.global_dim,
                "start_offset": self.start_offset,
                "end_offset": self.end_offset,
                "activation": keras.activations.serialize(self.activation),
                "n_heads": self.n_heads,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CrossAttentionLayer":
        config = _strip_legacy_keys(
            config,
            {
                "layer_norm00",
                "layer_norm01",
                "layer_norm1",
                "ffn",
                "add0",
                "add1",
                "attention",
            },
        )
        return cls(**config)

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        local_repr = self.layer_norm00(inputs[0])
        global_repr = self.layer_norm01(inputs[1])

        query_end = local_repr.shape[1] - self.end_offset if self.end_offset else None
        query = local_repr[:, self.start_offset:query_end, :]

        attention_output, attention_scores = self.attention(
            query,
            global_repr,
            global_repr,
            return_attention_scores=True,
            training=training,
        )
        attention_output = self.add0([attention_output, query])
        attention_output = self.layer_norm1(attention_output)
        outputs = self.ffn(attention_output, training=training)
        outputs = self.add1([outputs, attention_output])
        return outputs, attention_scores


@keras.utils.register_keras_serializable(package="ssSTICI", name="SelfAttentionBlock")
class SelfAttentionBlock(layers.Layer):
    """ssSTICI self-attention block.

    This is the architecture-level self-attention modification relative to the
    public STICI implementation: it applies LayerNorm before multi-head attention,
    then residual + LayerNorm, then FFN + residual.

    ``attention_range`` and ``dropout_rate`` are accepted only to load legacy
    checkpoints whose configs contained those keys; they are not used by this
    block.  Local context is controlled by the outer overlapping chunk windows.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        start_offset: int = 0,
        end_offset: int = 0,
        activation: Union[str, Dict[str, Any]] = "gelu",
        attention_range: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.start_offset = int(start_offset)
        self.end_offset = int(end_offset)
        self.activation = keras.activations.get(activation)

        self.att0 = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim,
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.ff_dim, activation=self.activation),
                layers.Dense(self.embed_dim, activation=self.activation),
            ],
            name="self_attention_ffn",
        )
        self.layer_norm0 = layers.LayerNormalization()
        self.layer_norm1 = layers.LayerNormalization()

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "start_offset": self.start_offset,
                "end_offset": self.end_offset,
                "activation": keras.activations.serialize(self.activation),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SelfAttentionBlock":
        config = _strip_legacy_keys(
            config,
            {"ffn", "att0", "layer_norm0", "layer_norm1"},
        )
        return cls(**config)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.layer_norm0(inputs)
        query_end = x.shape[1] - self.end_offset if self.end_offset else None
        query = x[:, self.start_offset:query_end, :]
        attn_output, attention_scores = self.att0(
            query,
            x,
            return_attention_scores=True,
            training=training,
        )
        out1 = self.layer_norm1(query + attn_output)
        ffn_output = self.ffn(out1, training=training)
        return out1 + ffn_output, attention_scores


@keras.utils.register_keras_serializable(package="ssSTICI")
class CatEmbeddings(layers.Layer):
    """Learned categorical base embedding plus learned positional embedding."""

    def __init__(
        self,
        embedding_dim: int,
        embeddings_initializer: Union[str, Dict[str, Any]] = "glorot_uniform",
        embeddings_regularizer: Optional[Union[str, Dict[str, Any]]] = None,
        activity_regularizer: Optional[Union[str, Dict[str, Any]]] = None,
        embeddings_constraint: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        activity_regularizer_obj = regularizers.get(activity_regularizer)
        super().__init__(activity_regularizer=activity_regularizer_obj, **kwargs)
        self.embedding_dim = int(embedding_dim)
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self._activity_regularizer_config = activity_regularizer_obj

    def build(self, input_shape: tf.TensorShape) -> None:
        num_alleles = int(input_shape[-1])
        n_sites = int(input_shape[-2])
        self.position_embedding = layers.Embedding(
            input_dim=n_sites,
            output_dim=self.embedding_dim,
            name="position_embedding",
        )
        self.embedding = self.add_weight(
            shape=(num_alleles, self.embedding_dim),
            initializer=self.embeddings_initializer,
            trainable=True,
            name="cat_embeddings",
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            experimental_autocast=False,
        )
        self.positions = tf.range(start=0, limit=n_sites, delta=1)
        super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "embeddings_initializer": initializers.serialize(
                    self.embeddings_initializer
                ),
                "embeddings_regularizer": regularizers.serialize(
                    self.embeddings_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self._activity_regularizer_config
                ),
                "embeddings_constraint": constraints.serialize(
                    self.embeddings_constraint
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CatEmbeddings":
        config = _strip_legacy_keys(
            config,
            {
                "position_embedding",
                "embedding",
                "positions",
                "num_of_allels",
                "n_snps",
            },
        )
        return cls(**config)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        immediate_result = tf.einsum("ijk,kl->ijl", inputs, self.embedding)
        return immediate_result + self.position_embedding(self.positions)


@keras.utils.register_keras_serializable(package="ssSTICI")
class SelfAttnChunk(layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        start_offset: int = 0,
        end_offset: int = 0,
        attention_range: Optional[int] = None,  # legacy compatibility
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.start_offset = int(start_offset)
        self.end_offset = int(end_offset)
        self.attention_block = SelfAttentionBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "start_offset": self.start_offset,
                "end_offset": self.end_offset,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SelfAttnChunk":
        config = _strip_legacy_keys(config, {"attention_block"})
        return cls(**config)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.attention_block(inputs, training=training)


@keras.utils.register_keras_serializable(package="ssSTICI")
class CrossAttnChunk(layers.Layer):
    def __init__(
        self,
        start_offset: int = 0,
        end_offset: int = 0,
        n_heads: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.start_offset = int(start_offset)
        self.end_offset = int(end_offset)
        self.n_heads = int(n_heads)

    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
        local_dim = int(input_shape[0][-1])
        global_dim = int(input_shape[1][-1])
        self.attention_block = CrossAttentionLayer(
            local_dim=local_dim,
            global_dim=global_dim,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
            n_heads=self.n_heads,
        )
        super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "start_offset": self.start_offset,
                "end_offset": self.end_offset,
                "n_heads": self.n_heads,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CrossAttnChunk":
        config = _strip_legacy_keys(
            config,
            {"local_dim", "global_dim", "attention_block"},
        )
        return cls(**config)

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.attention_block(inputs, training=training)


@keras.utils.register_keras_serializable(package="ssSTICI")
class ConvBlock(layers.Layer):
    """Integrated multi-kernel 1D convolution block retained from STICI."""

    def __init__(self, embed_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.conv000 = layers.Conv1D(
            self.embed_dim, 3, padding="same", activation="gelu"
        )
        self.conv010 = layers.Conv1D(
            self.embed_dim, 5, padding="same", activation="gelu"
        )
        self.conv011 = layers.Conv1D(
            self.embed_dim, 7, padding="same", activation="gelu"
        )
        self.conv020 = layers.Conv1D(
            self.embed_dim, 7, padding="same", activation="gelu"
        )
        self.conv021 = layers.Conv1D(
            self.embed_dim, 15, padding="same", activation="gelu"
        )
        self.add = layers.Add()
        self.conv100 = layers.Conv1D(
            self.embed_dim, 3, padding="same", activation="gelu"
        )
        self.bn0 = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.dw_conv = layers.Conv1D(self.embed_dim, 1, padding="same")
        self.activation = layers.Activation("gelu")

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvBlock":
        config = _strip_legacy_keys(
            config,
            {
                "const",
                "conv000",
                "conv010",
                "conv011",
                "conv020",
                "conv021",
                "add",
                "conv100",
                "bn0",
                "bn1",
                "dw_conv",
                "activation",
            },
        )
        return cls(**config)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        xa = self.conv000(inputs)

        xb = self.conv010(xa)
        xb = self.conv011(xb)

        xc = self.conv020(xa)
        xc = self.conv021(xc)

        xa = self.add([xb, xc])
        xa = self.conv100(xa)
        xa = self.bn0(xa, training=training)
        xa = self.dw_conv(xa)
        xa = self.bn1(xa, training=training)
        return self.activation(xa)


@keras.utils.register_keras_serializable(package="ssSTICI", name="chunk_module")
def chunk_module(
    input_len: int,
    embed_dim: int,
    num_heads: int,
    start_offset: int = 0,
    end_offset: int = 0,
    dropout_rate: float = 0.25,
    cross_attention_heads: int = 8,
) -> keras.Model:
    """Construct one internal ssSTICI chunk model."""
    projection_dim = int(embed_dim)
    inputs = layers.Input(shape=(int(input_len), projection_dim))

    xa0, self_attention_scores = SelfAttnChunk(
        embed_dim=projection_dim,
        num_heads=int(num_heads),
        ff_dim=projection_dim // 2,
        start_offset=int(start_offset),
        end_offset=int(end_offset),
    )(inputs)

    xa = ConvBlock(projection_dim)(xa0)
    xa_skip = ConvBlock(projection_dim)(xa)

    xa = layers.Dense(projection_dim, activation="gelu")(xa)
    xa = ConvBlock(projection_dim)(xa)
    xa, cross_attention_scores = CrossAttnChunk(
        start_offset=0,
        end_offset=0,
        n_heads=int(cross_attention_heads),
    )([xa, xa0])
    xa = layers.Dropout(float(dropout_rate))(xa)
    xa = ConvBlock(projection_dim)(xa)
    xa = layers.Concatenate(axis=-1)([xa_skip, xa])

    return keras.Model(
        inputs=inputs,
        outputs=[xa, self_attention_scores, cross_attention_scores],
        name="ssstici_internal_chunk",
    )


# -----------------------------------------------------------------------------
# ssSTICI model
# -----------------------------------------------------------------------------

@keras.utils.register_keras_serializable(package="ssSTICI", name="ssSTICI")
class SsSTICI(keras.Model):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        offset_before: int = 0,
        offset_after: int = 0,
        chunk_size: int = 2048,
        activation: Union[str, Dict[str, Any]] = "gelu",
        dropout_rate: float = 0.25,
        attention_range: int = 256,
        cross_attention_heads: int = 8,
        global_batch_size: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.offset_before = int(offset_before)
        self.offset_after = int(offset_after)
        self.chunk_size = int(chunk_size)
        self.activation = keras.activations.get(activation)
        self.dropout_rate = float(dropout_rate)
        self.attention_range = int(attention_range)
        self.cross_attention_heads = int(cross_attention_heads)
        self.global_batch_size = int(global_batch_size)

        self.cce_fn = keras.losses.CategoricalCrossentropy(
            reduction=keras.losses.Reduction.NONE
        )
        self.kld_fn = keras.losses.KLDivergence(
            reduction=keras.losses.Reduction.NONE
        )
        self.loss_tracker = keras.metrics.Mean(name="rec_loss")
        self.accuracy_tracker_seq = keras.metrics.CategoricalAccuracy(name="acc")

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        return [self.loss_tracker, self.accuracy_tracker_seq]

    def build(self, input_shape: tf.TensorShape) -> None:
        self.seq_len = int(input_shape[1])
        self.in_channel = int(input_shape[-1])
        self.masking_val = self.in_channel - 1

        self.chunk_starts = list(range(0, self.seq_len, self.chunk_size))
        self.chunk_ends = [
            min(start + self.chunk_size, self.seq_len)
            for start in self.chunk_starts
        ]
        self.mask_starts = [
            max(0, start - self.attention_range)
            for start in self.chunk_starts
        ]
        self.mask_ends = [
            min(end + self.attention_range, self.seq_len)
            for end in self.chunk_ends
        ]

        self.chunkers = []
        for i, start in enumerate(self.chunk_starts):
            self.chunkers.append(
                chunk_module(
                    input_len=self.mask_ends[i] - self.mask_starts[i],
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    start_offset=start - self.mask_starts[i],
                    end_offset=self.mask_ends[i] - self.chunk_ends[i],
                    dropout_rate=self.dropout_rate,
                    cross_attention_heads=self.cross_attention_heads,
                )
            )

        self.concat_layer = layers.Concatenate(axis=-2)
        self.embedding = CatEmbeddings(self.embed_dim)
        self.after_concat_layer = layers.Conv1D(
            self.embed_dim // 2,
            5,
            padding="same",
            activation=self.activation,
        )
        self.last_conv = layers.Conv1D(
            self.in_channel - 1,
            5,
            padding="same",
            activation="softmax",
        )
        super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "offset_before": self.offset_before,
                "offset_after": self.offset_after,
                "chunk_size": self.chunk_size,
                "activation": keras.activations.serialize(self.activation),
                "dropout_rate": self.dropout_rate,
                "attention_range": self.attention_range,
                "cross_attention_heads": self.cross_attention_heads,
                "global_batch_size": self.global_batch_size,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SsSTICI":
        config = _strip_legacy_keys(
            config,
            {
                "in_channel",
                "seq_len",
                "masking_val",
                "accuracy_tracker_seq",
                "loss_tracker",
                "cce_fn",
                "kld_fn",
                "chunk_starts",
                "chunk_ends",
                "mask_starts",
                "mask_ends",
                "chunkers",
                "embedding",
                "concat_layer",
                "after_concat_layer",
                "last_conv",
            },
        )
        config.setdefault("cross_attention_heads", 8)
        return cls(**config)

    def compute_ds_loss(
        self,
        loss_object: keras.losses.Loss,
        labels: tf.Tensor,
        predictions: tf.Tensor,
    ) -> tf.Tensor:
        per_position_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(
            per_position_loss,
            global_batch_size=self.global_batch_size,
        )

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
        x = self.embedding(inputs)
        chunks = [
            self.chunkers[i](
                x[:, self.mask_starts[i]:self.mask_ends[i], :],
                training=training,
            )
            for i in range(len(self.chunkers))
        ]

        x = self.concat_layer([chunk[0] for chunk in chunks])
        x = self.after_concat_layer(x)
        x = self.last_conv(x)

        output_end = self.seq_len - self.offset_after if self.offset_after else None
        x = x[:, self.offset_before:output_end, :]

        self_attentions = [tf.cast(chunk[1], tf.float16) for chunk in chunks]
        cross_attentions = [tf.cast(chunk[2], tf.float16) for chunk in chunks]
        return x, self_attentions, cross_attentions

    def get_observed_positions(self, values: tf.Tensor, ground_truth: tf.Tensor) -> tf.Tensor:
        observed_locations = tf.where(
            tf.not_equal(tf.argmax(ground_truth, axis=-1), self.masking_val)
        )
        return tf.gather_nd(values, observed_locations)

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        noisy_samples, target_samples = data
        real_target_samples = self.get_observed_positions(
            target_samples,
            target_samples,
        )[..., :-1]

        with tf.GradientTape() as tape:
            reconstructed_samples, _, _ = self(noisy_samples, training=True)
            real_reconstructed_samples = self.get_observed_positions(
                reconstructed_samples,
                target_samples,
            )
            rec_loss = self.compute_ds_loss(
                self.cce_fn,
                real_target_samples,
                real_reconstructed_samples,
            )
            rec_loss += self.compute_ds_loss(
                self.kld_fn,
                real_target_samples,
                real_reconstructed_samples,
            )
            if self.losses:
                rec_loss += tf.nn.scale_regularization_loss(tf.add_n(self.losses))

        gradients = tape.gradient(rec_loss, self.trainable_variables)
        gradient_variable_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.trainable_variables)
            if gradient is not None
        ]
        self.optimizer.apply_gradients(gradient_variable_pairs)

        self.loss_tracker.update_state(rec_loss)
        self.accuracy_tracker_seq.update_state(
            real_target_samples,
            real_reconstructed_samples,
        )
        return {
            "rec_loss": self.loss_tracker.result(),
            "acc": self.accuracy_tracker_seq.result(),
        }

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        noisy_samples, target_samples = data
        reconstructed_samples, _, _ = self(noisy_samples, training=False)

        real_target_samples = self.get_observed_positions(
            target_samples,
            target_samples,
        )[..., :-1]
        real_reconstructed_samples = self.get_observed_positions(
            reconstructed_samples,
            target_samples,
        )

        rec_loss = self.compute_ds_loss(
            self.cce_fn,
            real_target_samples,
            real_reconstructed_samples,
        )
        rec_loss += self.compute_ds_loss(
            self.kld_fn,
            real_target_samples,
            real_reconstructed_samples,
        )

        self.loss_tracker.update_state(rec_loss)
        self.accuracy_tracker_seq.update_state(
            real_target_samples,
            real_reconstructed_samples,
        )
        return {
            "rec_loss": self.loss_tracker.result(),
            "acc": self.accuracy_tracker_seq.result(),
        }

    def predict_step(self, data: tf.Tensor):
        return self(data, training=False)


# Legacy names are intentionally mapped to the cleaned implementations so older
# checkpoints have a best-effort load path. New checkpoints use ssSTICI names.
custom_objects = {
    "ssSTICI": SsSTICI,
    "SsSTICI": SsSTICI,
    "SplitTransformer": SsSTICI,
    "CrossAttentionLayer": CrossAttentionLayer,
    "SelfAttentionBlock": SelfAttentionBlock,
    "MaskedTransformerBlock": SelfAttentionBlock,
    "CatEmbeddings": CatEmbeddings,
    "GenoEmbeddings": CatEmbeddings,
    "SelfAttnChunk": SelfAttnChunk,
    "CrossAttnChunk": CrossAttnChunk,
    "ConvBlock": ConvBlock,
    "chunk_module": chunk_module,
    # Registered-name aliases emitted by the historical implementation.
    "MyModels>SplitTransformer": SsSTICI,
    "MyLayers>CrossAttentionLayer": CrossAttentionLayer,
    "MyLayers>MaskedTransformerBlock": SelfAttentionBlock,
    "MyLayers>CatEmbeddings": CatEmbeddings,
    "MyLayers>GenoEmbeddings": CatEmbeddings,
    "MyLayers>SelfAttnChunk": SelfAttnChunk,
    "MyLayers>CrossAttnChunk": CrossAttnChunk,
    "MyLayers>ConvBlock": ConvBlock,
    "MyLayers>chunk_module": chunk_module,
}


# -----------------------------------------------------------------------------
# Model creation and training helpers
# -----------------------------------------------------------------------------

def create_model(model_args: Dict[str, Any]) -> SsSTICI:
    model = SsSTICI(
        embed_dim=model_args["embedding_dim"],
        num_heads=model_args["num_heads"],
        chunk_size=model_args["chunk_size"],
        activation="gelu",
        dropout_rate=model_args["dropout_rate"],
        attention_range=model_args["chunk_overlap"],
        cross_attention_heads=model_args["cross_attention_heads"],
        offset_before=model_args["offset_before"],
        offset_after=model_args["offset_after"],
        global_batch_size=model_args["global_batch_size"],
        name=MODEL_NAME,
    )
    optimizer = tfa.optimizers.LAMB(learning_rate=model_args["lr"])
    model.compile(optimizer=optimizer, weighted_metrics=[])
    return model


def create_callbacks(metric: str = "rec_loss") -> List[keras.callbacks.Callback]:
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor=metric,
            mode="auto",
            factor=0.5,
            patience=5,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor=metric,
            mode="auto",
            patience=15,
            verbose=1,
            restore_best_weights=True,
        ),
    ]


class DataReader:
    def __init__(self) -> None:
        self.target_set: Optional[np.ndarray] = None
        self.target_ids: List[str] = []
        self.SITE_COUNT = 0
        self.reference_panel: Optional[np.ndarray] = None
        self.ALPHA = ALPHABET
        self.MISSING_VALUE = MISSING_VALUE
        self.SEQ_DEPTH = SEQUENCE_DEPTH

    def map_nucleotides(self, sequence: str) -> str:
        sequence = sequence.upper()
        return "".join(char if char in self.ALPHA else MISSING_SYMBOL for char in sequence)

    def _read_fasta(self, file_path: Union[str, Path]) -> Tuple[List[str], np.ndarray]:
        """Read standard single- or multi-line FASTA and return encoded sequences."""
        file_path = str(file_path)
        pprint(f"Reading FASTA: {file_path}")
        opener = gzip.open if file_path.endswith(".gz") else open

        sample_ids: List[str] = []
        sequences: List[str] = []
        current_id: Optional[str] = None
        current_sequence: List[str] = []

        with opener(file_path, "rt") as fin:
            for raw_line in fin:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id is not None:
                        sequences.append("".join(current_sequence))
                    current_id = line
                    sample_ids.append(line)
                    current_sequence = []
                else:
                    if current_id is None:
                        raise ValueError(
                            f"Invalid FASTA {file_path}: sequence data appeared before the first header."
                        )
                    current_sequence.append(line)

        if current_id is not None:
            sequences.append("".join(current_sequence))

        if not sample_ids:
            raise ValueError(f"No FASTA records found in {file_path}.")
        if len(sample_ids) != len(sequences):
            raise ValueError(
                f"FASTA parsing mismatch in {file_path}: {len(sample_ids)} headers but "
                f"{len(sequences)} sequences."
            )

        mapped_sequences = [self.map_nucleotides(seq) for seq in sequences]
        lengths = {len(seq) for seq in mapped_sequences}
        if len(lengths) != 1:
            raise ValueError(
                f"All ssSTICI FASTA records must contain the same number of sites; "
                f"found lengths {sorted(lengths)} in {file_path}."
            )
        if next(iter(lengths)) == 0:
            raise ValueError(f"Empty FASTA sequence found in {file_path}.")

        missing_rates = [seq.count(MISSING_SYMBOL) / len(seq) for seq in mapped_sequences]
        pprint(
            "Missing rate — "
            f"max: {max(missing_rates):.6f}, min: {min(missing_rates):.6f}, "
            f"mean: {np.mean(missing_rates):.6f}, "
            f"25th percentile: {np.percentile(missing_rates, 25):.6f}, "
            f"75th percentile: {np.percentile(missing_rates, 75):.6f}"
        )

        encoded = np.asarray(
            [[self.ALPHA.index(base) for base in seq] for seq in mapped_sequences],
            dtype=np.int32,
        )
        return sample_ids, encoded

    def assign_training_set(self, file_path: str) -> None:
        _, self.reference_panel = self._read_fasta(file_path)
        self.SITE_COUNT = int(self.reference_panel.shape[1])
        pprint(
            f"Training matrix: {self.reference_panel.shape[0]} cells x "
            f"{self.SITE_COUNT} variant sites."
        )

    def assign_test_set(self, file_path: str, expected_site_count: Optional[int] = None) -> None:
        self.target_ids, self.target_set = self._read_fasta(file_path)
        site_count = int(self.target_set.shape[1])
        if expected_site_count is not None and site_count != int(expected_site_count):
            raise ValueError(
                f"Target contains {site_count} sites, but the trained ssSTICI run expects "
                f"{int(expected_site_count)} sites in the same order."
            )
        pprint(f"Target matrix: {self.target_set.shape[0]} cells x {site_count} variant sites.")

    def get_ref_set(self, start: int = 0, end: int = 0) -> np.ndarray:
        if self.reference_panel is None:
            raise RuntimeError("Training set has not been assigned.")
        if 0 <= start < end:
            return self.reference_panel[:, start:end]
        return self.reference_panel

    def get_target_set(self, start: int = 0, end: int = 0) -> np.ndarray:
        if self.target_set is None:
            raise RuntimeError("Target set has not been assigned.")
        if 0 <= start < end:
            return self.target_set[:, start:end]
        return self.target_set

    def idx_to_alpha(
        self,
        probability_array: np.ndarray,
        confidence_score_threshold: float = 0.0,
    ) -> List[str]:
        sequences: List[str] = []
        for sample_probs in tqdm(probability_array, desc="Converting probabilities to bases"):
            sequence = []
            for site_probs in sample_probs:
                if float(np.max(site_probs)) <= confidence_score_threshold:
                    sequence.append("-")
                else:
                    sequence.append(self.ALPHA[int(np.argmax(site_probs))])
            sequences.append("".join(sequence))
        return sequences

    def preds_to_bps(
        self,
        predictions: Union[str, np.ndarray],
        confidence_score_threshold: float,
    ) -> List[str]:
        preds = np.load(predictions) if isinstance(predictions, str) else predictions
        return self.idx_to_alpha(preds, confidence_score_threshold)

    def write_to_fasta(
        self,
        out_file_path: Union[str, Path],
        pred_seqs: Union[str, np.ndarray],
        confidence_score_threshold: float = 0.0,
    ) -> Path:
        output_path = Path(f"{out_file_path}.fasta")
        seqs = self.preds_to_bps(pred_seqs, confidence_score_threshold)
        with open(output_path, "w") as fout:
            for i, sequence in enumerate(tqdm(seqs, desc="Writing FASTA")):
                sample_id = self.target_ids[i] if self.target_ids else f">sample_{i + 1}"
                fout.write(f"{sample_id}\n{sequence}\n")
        return output_path

    def write_top_probabilities_to_tsv(
        self,
        out_file_path: Union[str, Path],
        pred_seqs: Union[str, np.ndarray],
    ) -> Path:
        preds = np.load(pred_seqs) if isinstance(pred_seqs, str) else pred_seqs
        # preds has four channels (A/T/G/C); therefore the missing class is
        # intrinsically excluded from this maximum.
        top_probabilities = np.max(preds, axis=-1)
        output_path = Path(f"{out_file_path}.tsv")
        with open(output_path, "w") as fout:
            fout.write(
                "sample_id\t"
                + "\t".join(
                    f"site_{site + 1}" for site in range(top_probabilities.shape[1])
                )
                + "\n"
            )
            for i, sample_probabilities in enumerate(top_probabilities):
                sample_id = (
                    self.target_ids[i].lstrip(">")
                    if self.target_ids
                    else f"sample_{i + 1}"
                )
                fout.write(
                    sample_id
                    + "\t"
                    + "\t".join(str(float(p)) for p in sample_probabilities)
                    + "\n"
                )
        return output_path


def create_directories(save_dir: Union[str, Path]) -> None:
    save_dir = Path(save_dir)
    (save_dir / "models").mkdir(parents=True, exist_ok=True)
    (save_dir / "out").mkdir(parents=True, exist_ok=True)


def clear_dir(path: Union[str, Path]) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def chunk_breakpoints(site_count: int, sites_per_model: int) -> List[int]:
    break_points = list(range(0, int(site_count), int(sites_per_model)))
    if not break_points or break_points[-1] != int(site_count):
        break_points.append(int(site_count))
    return break_points


def load_chunk_info(save_dir: Union[str, Path], number_of_chunks: int) -> Dict[int, bool]:
    expected = {i: False for i in range(number_of_chunks)}
    path = Path(save_dir) / "models" / "chunks_info.json"
    if not path.is_file():
        return expected
    with open(path) as fin:
        loaded = json.load(fin)
    if not isinstance(loaded, dict) or len(loaded) != number_of_chunks:
        warn("Ignoring chunks_info.json because its chunk count does not match this run.")
        return expected
    pprint("Resuming ssSTICI training from chunk status file.")
    return {int(key): bool(value) for key, value in loaded.items()}


def save_chunk_status(save_dir: Union[str, Path], chunk_info: Dict[int, bool]) -> None:
    path = Path(save_dir) / "models" / "chunks_info.json"
    with open(path, "w") as fout:
        json.dump(chunk_info, fout, indent=2, sort_keys=True)


def model_path(save_dir: Union[str, Path], chunk_index: int) -> Path:
    return Path(save_dir) / "models" / f"ssstici_chunk_{chunk_index + 1:03d}.keras"


def legacy_model_path(save_dir: Union[str, Path], chunk_index: int) -> Path:
    return Path(save_dir) / "models" / f"w_{chunk_index}.ckpt"


def load_model_for_inference(save_dir: Union[str, Path], chunk_index: int) -> keras.Model:
    path = model_path(save_dir, chunk_index)
    if not path.exists():
        legacy = legacy_model_path(save_dir, chunk_index)
        if legacy.exists():
            warn(f"Loading legacy checkpoint format: {legacy}")
            path = legacy
        else:
            raise FileNotFoundError(
                f"No trained model found for chunk {chunk_index + 1}: expected {path} "
                f"or legacy path {legacy}."
            )
    return keras.models.load_model(
        path,
        custom_objects=custom_objects,
        compile=False,
    )


@tf.function
def add_attention_mask(
    x_sample: tf.Tensor,
    y_sample: tf.Tensor,
    depth: int,
    masking_rate: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Mask a fraction of currently observed bases in the model input.

    The target remains the original sequence slice. Positions that were already
    missing in the target are subsequently excluded from the loss.
    """
    mask_value = tf.cast(depth - 1, tf.int32)
    observed_locations = tf.where(tf.not_equal(x_sample, mask_value))
    mask_size = tf.cast(
        tf.cast(tf.shape(observed_locations)[0], tf.float32) * masking_rate,
        tf.int32,
    )
    shuffled_locations = tf.random.shuffle(observed_locations)
    mask_locations = shuffled_locations[:mask_size]
    updates = tf.fill([tf.shape(mask_locations)[0]], mask_value)
    masked_input = tf.tensor_scatter_nd_update(x_sample, mask_locations, updates)
    return tf.one_hot(masked_input, depth), tf.one_hot(y_sample, depth)


def get_training_dataset(
    x: np.ndarray,
    batch_size: int,
    depth: int,
    strategy: tf.distribute.Strategy,
    offset_before: int = 0,
    offset_after: int = 0,
    masking_rate: float = 0.8,
    random_seed: int = 2024,
) -> tf.data.Dataset:
    auto = tf.data.AUTOTUNE
    target_end = x.shape[1] - offset_after if offset_after else None
    dataset = tf.data.Dataset.from_tensor_slices(
        (x, x[:, offset_before:target_end])
    )
    dataset = dataset.shuffle(
        buffer_size=x.shape[0],
        seed=int(random_seed),
        reshuffle_each_iteration=True,
    )
    dataset = dataset.repeat()
    dataset = dataset.map(
        lambda xx, yy: add_attention_mask(
            xx,
            yy,
            depth,
            masking_rate,
        ),
        num_parallel_calls=auto,
        deterministic=True,
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(auto)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.FILE
    )
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    return strategy.experimental_distribute_dataset(dataset)


RUN_METADATA_NAME = "run_metadata.json"
LEGACY_ARGS_NAME = "commandline_args.json"


def run_metadata_path(save_dir: Union[str, Path]) -> Path:
    return Path(save_dir) / RUN_METADATA_NAME


def write_run_metadata(
    args: argparse.Namespace,
    dr: DataReader,
    break_points: List[int],
    num_replicas: int,
    global_batch_size: int,
    steps_per_epoch: int,
) -> Dict[str, Any]:
    metadata = {
        "model_name": MODEL_NAME,
        "code_version": CODE_VERSION,
        "command": " ".join(shlex.quote(token) for token in sys.argv),
        "code": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "software_versions": software_versions(),
        "random_seed": int(args.random_seed),
        "deterministic_ops": bool(args.deterministic_ops),
        "data": {
            "training_fasta": str(Path(args.ref).resolve()),
            "training_fasta_sha256": sha256_file(args.ref),
            "sample_count": int(dr.reference_panel.shape[0]),
            "site_count": int(dr.SITE_COUNT),
            "alphabet": ALPHABET,
            "missing_symbol": MISSING_SYMBOL,
        },
        "training_arguments": vars(args).copy(),
        "derived_training": {
            "num_replicas": int(num_replicas),
            "global_batch_size": int(global_batch_size),
            "steps_per_epoch": int(steps_per_epoch),
            "break_points": [int(value) for value in break_points],
            "number_of_outer_models": len(break_points) - 1,
            "outer_model_seed_rule": "random_seed + zero_based_outer_model_index",
        },
    }
    with open(run_metadata_path(args.save_dir), "w") as fout:
        json.dump(metadata, fout, indent=2, sort_keys=True)

    # Keep the historical filename as a convenience for downstream scripts.
    with open(Path(args.save_dir) / LEGACY_ARGS_NAME, "w") as fout:
        json.dump(vars(args), fout, indent=2, sort_keys=True)
    return metadata


REPRODUCIBILITY_ARGS = (
    "ref",
    "sites_per_model",
    "co",
    "cs",
    "mr",
    "random_seed",
    "epochs",
    "na_heads",
    "cross_attention_heads",
    "embed_dim",
    "dropout_rate",
    "lr",
    "batch_size_per_gpu",
    "deterministic_ops",
)


def validate_resume_configuration(
    args: argparse.Namespace,
    metadata: Dict[str, Any],
    current_ref_sha256: str,
    global_batch_size: int,
    num_replicas: int,
) -> None:
    prior = metadata.get("training_arguments", {})
    mismatches = []
    for key in REPRODUCIBILITY_ARGS:
        if key == "ref":
            continue
        if key in prior and getattr(args, key, None) != prior[key]:
            mismatches.append(f"{key}: previous={prior[key]!r}, current={getattr(args, key)!r}")

    prior_hash = metadata.get("data", {}).get("training_fasta_sha256")
    if prior_hash and current_ref_sha256 != prior_hash:
        mismatches.append("training FASTA SHA256 differs from the existing run")

    derived = metadata.get("derived_training", {})
    if derived.get("global_batch_size") not in (None, int(global_batch_size)):
        mismatches.append(
            f"global_batch_size: previous={derived.get('global_batch_size')}, "
            f"current={global_batch_size}"
        )
    if derived.get("num_replicas") not in (None, int(num_replicas)):
        mismatches.append(
            f"num_replicas: previous={derived.get('num_replicas')}, current={num_replicas}"
        )

    if mismatches:
        raise ValueError(
            "Refusing to resume a ssSTICI run with training settings that differ "
            "from the saved metadata:\n  - " + "\n  - ".join(mismatches)
        )


def load_training_metadata(save_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    path = run_metadata_path(save_dir)
    if path.is_file():
        with open(path) as fin:
            return json.load(fin)
    legacy_path = Path(save_dir) / LEGACY_ARGS_NAME
    if legacy_path.is_file():
        with open(legacy_path) as fin:
            return {
                "model_name": "legacy-ssSTICI",
                "training_arguments": json.load(fin),
            }
    return None


def train_the_model(args: argparse.Namespace) -> None:
    seed_everything(args.random_seed, args.deterministic_ops)

    if args.restart_training:
        clear_dir(args.save_dir)
    create_directories(args.save_dir)

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice()
    )
    num_replicas = int(strategy.num_replicas_in_sync)
    global_batch_size = int(args.batch_size_per_gpu) * num_replicas
    pprint(f"Visible replicas used by MirroredStrategy: {num_replicas}")
    pprint(f"Global batch size: {global_batch_size}")

    dr = DataReader()
    dr.assign_training_set(args.ref)
    sample_count = int(dr.reference_panel.shape[0])
    if sample_count < global_batch_size:
        raise ValueError(
            f"Training matrix has {sample_count} cells, but global batch size is "
            f"{global_batch_size}. Reduce --batch-size-per-gpu or visible GPU count."
        )
    steps_per_epoch = sample_count // global_batch_size

    break_points = chunk_breakpoints(dr.SITE_COUNT, args.sites_per_model)
    number_of_chunks = len(break_points) - 1

    existing_metadata = load_training_metadata(args.save_dir)
    current_hash = sha256_file(args.ref)
    if existing_metadata and not args.restart_training and run_metadata_path(args.save_dir).is_file():
        validate_resume_configuration(
            args,
            existing_metadata,
            current_hash,
            global_batch_size,
            num_replicas,
        )
    else:
        write_run_metadata(
            args,
            dr,
            break_points,
            num_replicas,
            global_batch_size,
            steps_per_epoch,
        )

    chunks_done = load_chunk_info(args.save_dir, number_of_chunks)

    for w in range(number_of_chunks):
        if chunks_done[w]:
            pprint(f"Skipping outer model {w + 1}/{number_of_chunks}: already trained.")
            continue
        if args.which_chunk != -1 and w + 1 != args.which_chunk:
            pprint(
                f"Skipping outer model {w + 1}/{number_of_chunks} because "
                f"--which-chunk={args.which_chunk}."
            )
            continue

        pprint(f"Training outer model {w + 1}/{number_of_chunks}")
        # Make each saved outer model reproducible independently, including when
        # training is resumed at a later chunk.
        chunk_seed = int(args.random_seed) + w
        random.seed(chunk_seed)
        np.random.seed(chunk_seed)
        tf.keras.utils.set_random_seed(chunk_seed)
        pprint(f"Outer model random seed: {chunk_seed}")

        final_start_pos = max(0, break_points[w] - 2 * args.co)
        final_end_pos = min(dr.SITE_COUNT, break_points[w + 1] + 2 * args.co)
        offset_before = break_points[w] - final_start_pos
        offset_after = final_end_pos - break_points[w + 1]

        ref_set = dr.get_ref_set(final_start_pos, final_end_pos).astype(np.int32)
        pprint(
            f"Training window sites [{final_start_pos}, {final_end_pos}); "
            f"shape={ref_set.shape}, output offsets=({offset_before}, {offset_after})"
        )

        train_dataset = get_training_dataset(
            ref_set,
            batch_size=global_batch_size,
            depth=dr.SEQ_DEPTH,
            strategy=strategy,
            offset_before=offset_before,
            offset_after=offset_after,
            masking_rate=args.mr,
            random_seed=chunk_seed,
        )
        build_shape = [global_batch_size, ref_set.shape[1], dr.SEQ_DEPTH]
        del ref_set

        K.clear_session()
        with strategy.scope():
            model_args = {
                "embedding_dim": args.embed_dim,
                "num_heads": args.na_heads,
                "cross_attention_heads": args.cross_attention_heads,
                "chunk_size": args.cs,
                "chunk_overlap": args.co,
                "dropout_rate": args.dropout_rate,
                "offset_before": offset_before,
                "offset_after": offset_after,
                "lr": args.lr,
                "global_batch_size": global_batch_size,
            }
            model = create_model(model_args)
            model.build(build_shape)
            model.fit(
                train_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=args.epochs,
                callbacks=create_callbacks(),
                verbose=args.verbose,
            )
            output_model_path = model_path(args.save_dir, w)
            model.save(output_model_path)
            pprint(f"Saved {MODEL_NAME} model: {output_model_path}")

        chunks_done[w] = True
        save_chunk_status(args.save_dir, chunks_done)
        del model
        K.clear_session()


def resolve_imputation_layout(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], int, int, int]:
    metadata = load_training_metadata(args.save_dir)
    if metadata is None:
        raise FileNotFoundError(
            f"No {RUN_METADATA_NAME} or legacy {LEGACY_ARGS_NAME} found in {args.save_dir}."
        )

    training_args = metadata.get("training_arguments", {})
    for key in ("sites_per_model", "co", "cs"):
        if key not in training_args:
            raise KeyError(f"Saved training metadata is missing required key {key!r}.")

    sites_per_model = int(training_args["sites_per_model"])
    co = int(training_args["co"])
    cs = int(training_args["cs"])
    return metadata, sites_per_model, co, cs


def impute_the_target(args: argparse.Namespace) -> None:
    if not args.target:
        raise ValueError("--target is required in impute mode.")

    metadata, sites_per_model, co, _ = resolve_imputation_layout(args)
    training_seed = int(
        metadata.get("random_seed", metadata.get("training_arguments", {}).get("random_seed", 2024))
    )
    seed_everything(training_seed, bool(metadata.get("deterministic_ops", False)))

    expected_site_count = metadata.get("data", {}).get("site_count")
    if expected_site_count is None:
        if not args.ref:
            raise ValueError(
                "This is a legacy run without saved site_count metadata. Supply --ref "
                "during imputation so ssSTICI can recover the training site count."
            )
        ref_reader = DataReader()
        ref_reader.assign_training_set(args.ref)
        expected_site_count = ref_reader.SITE_COUNT

    if args.ref:
        ref_reader = DataReader()
        ref_reader.assign_training_set(args.ref)
        if ref_reader.SITE_COUNT != int(expected_site_count):
            raise ValueError(
                f"--ref contains {ref_reader.SITE_COUNT} sites but the trained run expects "
                f"{expected_site_count}."
            )

    dr = DataReader()
    dr.assign_test_set(args.target, expected_site_count=int(expected_site_count))

    # Prediction is intentionally controlled by per-process batch size; unlike
    # training, model.predict is not wrapped in a distributed dataset here.
    batch_size = int(args.batch_size_per_gpu)

    break_points = chunk_breakpoints(int(expected_site_count), sites_per_model)
    all_preds: List[np.ndarray] = []

    for w in range(len(break_points) - 1):
        pprint(f"Imputing outer model {w + 1}/{len(break_points) - 1}")
        final_start_pos = max(0, break_points[w] - 2 * co)
        final_end_pos = min(int(expected_site_count), break_points[w + 1] + 2 * co)
        test_dataset_np = dr.get_target_set(final_start_pos, final_end_pos).astype(np.int32)

        K.clear_session()

        # Load outside any MirroredStrategy scope.  This is required for robust
        # Keras 2.14 .keras deserialization of this subclassed model (see note
        # above).  The direct model calls below still use an available GPU via
        # TensorFlow's normal device placement.
        model = load_model_for_inference(args.save_dir, w)
        batch_predictions: List[np.ndarray] = []

        for start_idx in tqdm(
            range(0, test_dataset_np.shape[0], batch_size),
            desc=f"Predicting chunk {w + 1}",
        ):
            end_idx = min(start_idx + batch_size, test_dataset_np.shape[0])
            one_hot = to_categorical(
                test_dataset_np[start_idx:end_idx],
                num_classes=dr.SEQ_DEPTH,
            ).astype(np.float32)
            if one_hot.shape[0] == 0:
                continue
            pred, _, _ = model(tf.convert_to_tensor(one_hot), training=False)
            batch_predictions.append(pred.numpy().astype(np.float32))

        if not batch_predictions:
            raise RuntimeError(f"No predictions were produced for outer model {w + 1}.")
        all_preds.append(np.vstack(batch_predictions))
        del model
        K.clear_session()

    predictions = np.hstack(all_preds)
    if predictions.shape[1] != int(expected_site_count):
        raise RuntimeError(
            f"Concatenated prediction matrix has {predictions.shape[1]} sites; "
            f"expected {expected_site_count}."
        )

    out_dir = Path(args.save_dir) / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = dr.write_to_fasta(
        out_dir / f"predictions_confidence_{args.confidence_threshold:.1f}",
        predictions,
        args.confidence_threshold,
    )
    probability_path = dr.write_top_probabilities_to_tsv(
        out_dir / "top_probability_per_site",
        predictions,
    )

    imputation_metadata = {
        "model_name": MODEL_NAME,
        "code_version": CODE_VERSION,
        "code": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "software_versions": software_versions(),
        "target_fasta": str(Path(args.target).resolve()),
        "target_fasta_sha256": sha256_file(args.target),
        "target_sample_count": int(predictions.shape[0]),
        "site_count": int(predictions.shape[1]),
        "confidence_threshold": float(args.confidence_threshold),
        "batch_size_per_gpu": int(args.batch_size_per_gpu),
        "training_metadata_file": str(run_metadata_path(args.save_dir)),
        "outputs": {
            "predicted_fasta": str(fasta_path),
            "top_probability_tsv": str(probability_path),
        },
    }
    with open(out_dir / "imputation_metadata.json", "w") as fout:
        json.dump(imputation_metadata, fout, indent=2, sort_keys=True)

    pprint(f"Done. Outputs are in {out_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ssSTICI: train a dataset-specific Split-Transformer with Integrated "
            "Convolutions on a sparse cell-variant FASTA matrix, or impute with "
            "a previously trained ssSTICI run."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["train", "impute"],
        default="train",
        help="Operation mode (default: train).",
    )
    parser.add_argument(
        "--restart-training",
        type=str_to_bool,
        default=False,
        help="Delete --save-dir before training and start from scratch (default: false).",
    )

    # Inputs and outputs
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help=(
            "Training FASTA in train mode. Optional in impute mode for validation; "
            "legacy runs without run_metadata.json require it."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target FASTA; required in impute mode.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory used to save/load ssSTICI models, metadata, and outputs.",
    )
    parser.add_argument(
        "--which-chunk",
        type=int,
        default=-1,
        help="Train only this 1-based outer model chunk; -1 trains all pending chunks.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=(
            "During FASTA writing, emit '-' when the maximum A/T/G/C probability is "
            "<= this threshold (default: 0.0)."
        ),
    )

    # Chunking
    parser.add_argument(
        "--co",
        type=int,
        default=256,
        help="Overlap/context size in sites (default: 256).",
    )
    parser.add_argument(
        "--cs",
        type=int,
        default=2048,
        help="Internal transformer chunk size in sites (default: 2048).",
    )
    parser.add_argument(
        "--sites-per-model",
        type=int,
        default=16000,
        help="Number of central output sites handled by each outer saved model (default: 16000).",
    )

    # Training/model parameters
    parser.add_argument(
        "--mr",
        type=float,
        default=0.8,
        help=(
            "Fraction of currently observed bases masked in each training example "
            "before reconstruction (default: 0.8)."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2024,
        help="Seed for Python, NumPy, TensorFlow, and tf.data shuffling (default: 2024).",
    )
    parser.add_argument(
        "--deterministic-ops",
        type=str_to_bool,
        default=False,
        help="Request deterministic TensorFlow operations where supported (default: false).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Maximum epochs per outer model (default: 1000).",
    )
    parser.add_argument(
        "--na-heads",
        "--num-heads",
        dest="na_heads",
        type=int,
        default=16,
        help="Number of self-attention heads (default: 16).",
    )
    parser.add_argument(
        "--cross-attention-heads",
        type=int,
        default=8,
        help="Number of cross-attention heads; historical ssSTICI value is 8 (default: 8).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Embedding dimension (default: 128).",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.25,
        help="Dropout after cross-attention inside each internal chunk (default: 0.25).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        help="LAMB learning rate (default: 0.002).",
    )
    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=8,
        help="Training batch size per visible GPU; inference batch size per process (default: 8).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Keras training verbosity (default: 2).",
    )

    args = parser.parse_args()

    if not (0.0 <= args.mr <= 1.0):
        parser.error("--mr must be between 0 and 1.")
    if not (0.0 <= args.dropout_rate < 1.0):
        parser.error("--dropout-rate must be in [0, 1).")
    if not (0.0 <= args.confidence_threshold <= 1.0):
        parser.error("--confidence-threshold must be between 0 and 1.")
    for name in ("co", "cs", "sites_per_model", "epochs", "na_heads", "cross_attention_heads", "embed_dim", "batch_size_per_gpu"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0.")

    if args.mode == "train" and not args.ref:
        parser.error("--ref is required in train mode.")
    if args.mode == "impute" and not args.target:
        parser.error("--target is required in impute mode.")

    if not (args.save_dir.startswith("./") or args.save_dir.startswith("/")):
        args.save_dir = f"./{args.save_dir}"
    return args


def main() -> None:
    args = parse_args()
    pprint(f"Save directory: {args.save_dir}")
    if args.mode == "train":
        train_the_model(args)
    else:
        impute_the_target(args)


if __name__ == "__main__":
    main()
