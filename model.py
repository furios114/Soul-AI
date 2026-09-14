import math

import torch
import torch.nn as nn


# ============================================================
# SOUL CONFIG
# ============================================================

class SoulConfig:

    def __init__(
        self,
        vocab_size,
        block_size=128,
        n_layers=4,
        n_heads=4,
        d_model=256,
        dropout=0.1
    ):

        self.vocab_size = vocab_size
        self.block_size = block_size

        self.n_layers = n_layers
        self.n_heads = n_heads

        self.d_model = d_model
        self.dropout = dropout

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model должен делиться на n_heads"
            )

        self.head_dim = d_model // n_heads
# ============================================================
# SOUL ATTENTION
# ============================================================

class SoulAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.head_dim

        # Q, K, V
        self.qkv = nn.Linear(
            config.d_model,
            config.d_model * 3
        )

        # Выходной слой
        self.out = nn.Linear(
            config.d_model,
            config.d_model
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        # Causal mask
        mask = torch.tril(
            torch.ones(
                config.block_size,
                config.block_size
            )
        )

        self.register_buffer(
            "mask",
            mask.view(
                1,
                1,
                config.block_size,
                config.block_size
            )
        )

    def forward(self, x):

        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1
        )

        # Разделяем на головы
        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        # Attention scores
        attention = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Запрещаем смотреть в будущее
        attention = attention.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf")
        )

        attention = torch.softmax(
            attention,
            dim=-1
        )

        attention = self.dropout(
            attention
        )

        # Применяем attention к V
        out = attention @ v

        # Возвращаем головы обратно
        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out.view(
            batch_size,
            seq_len,
            d_model
        )

        return self.out(out)
# ============================================================
# SOUL FEED FORWARD
# ============================================================

class SoulFeedForward(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                config.d_model,
                config.d_model * 4
            ),

            nn.GELU(),

            nn.Linear(
                config.d_model * 4,
                config.d_model
            ),

            nn.Dropout(
                config.dropout
            )
        )

    def forward(self, x):

        return self.network(x)
# ============================================================
# SOUL BLOCK
# ============================================================

class SoulBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            config.d_model
        )

        self.attention = SoulAttention(
            config
        )

        self.norm2 = nn.LayerNorm(
            config.d_model
        )

        self.feed_forward = SoulFeedForward(
            config
        )

    def forward(self, x):

        # Attention + residual
        x = x + self.attention(
            self.norm1(x)
        )

        # Feed Forward + residual
        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x
# ============================================================
# SOUL TRANSFORMER
# ============================================================

class SoulTransformer(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.d_model
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                SoulBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        # Language model head
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Инициализация весов
        self.apply(
            self._initialize_weights
        )

    def _initialize_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        input_ids,
        targets=None
    ):

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.block_size:

            raise ValueError(
                "Слишком длинная последовательность"
            )

        # Позиции
        positions = torch.arange(
            0,
            seq_len,
            device=input_ids.device
        )

        # Embeddings
        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = (
            token_embeddings
            + position_embeddings
        )

        # Transformer
        for block in self.blocks:

            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Logits
        logits = self.lm_head(x)

        loss = None

        # Если передали правильные ответы
        if targets is not None:

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss
# ============================================================
# GENERATION
# ============================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None
    ):

        self.eval()

        for _ in range(max_new_tokens):

            # Ограничиваем контекст
            input_context = input_ids[
                :, -self.config.block_size:
            ]

            # Получаем logits
            logits, _ = self(
                input_context
            )

            # Берём последний токен
            logits = logits[:, -1, :]

            # Temperature
            if temperature <= 0:

                temperature = 1.0

            logits = logits / temperature

            # Top-K
            if top_k is not None:

                top_k = min(
                    top_k,
                    logits.size(-1)
                )

                values, _ = torch.topk(
                    logits,
                    top_k
                )

                min_value = values[
                    :, -1
                ].unsqueeze(-1)

                logits = torch.where(
                    logits < min_value,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            # Вероятности
            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            # Выбираем следующий токен
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Добавляем его
            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

        return input_ids
# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SOUL AI - MODEL TEST")
    print("=" * 60)

    # -----------------------------------------------
    # Создаём конфигурацию
    # -----------------------------------------------

    config = SoulConfig(
        vocab_size=100,
        block_size=64,
        n_layers=4,
        n_heads=4,
        d_model=128,
        dropout=0.1
    )

    # -----------------------------------------------
    # Создаём модель
    # -----------------------------------------------

    model = SoulTransformer(
        config
    )

    # -----------------------------------------------
    # Количество параметров
    # -----------------------------------------------

    parameters = count_parameters(
        model
    )

    print(
        f"Количество параметров: {parameters:,}"
    )

    # -----------------------------------------------
    # Тестовый input
    # -----------------------------------------------

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 10)
    )

    print(
        f"Размер input: {input_ids.shape}"
    )

    # -----------------------------------------------
    # Forward
    # -----------------------------------------------

    logits, loss = model(
        input_ids
    )

    print(
        f"Размер logits: {logits.shape}"
    )

    # -----------------------------------------------
    # Generation
    # -----------------------------------------------

    generated = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=1.0,
        top_k=20
    )

    print(
        f"Размер generated: {generated.shape}"
    )

    # -----------------------------------------------
    # Результат
    # -----------------------------------------------

    print()
    print("Модель Soul успешно создана.")
    print("Forward pass работает.")
    print("Generation работает.")

    print("=" * 60)