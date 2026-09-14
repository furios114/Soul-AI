# ============================================================
# SOUL AI - TRAINING
# ============================================================

import os
import torch
import torch.nn as nn

from tokenizer import SoulTokenizer
from model import SoulConfig, SoulTransformer


# ============================================================
# НАСТРОЙКИ
# ============================================================

DATA_FILE = "data.txt"

MODEL_FILE = "soul_model.pt"
TOKENIZER_FILE = "tokenizer.json"

VOCAB_SIZE = 5000
BLOCK_SIZE = 64

N_LAYERS = 4
N_HEADS = 4
D_MODEL = 128

DROPOUT = 0.1

BATCH_SIZE = 16
LEARNING_RATE = 3e-4

EPOCHS = 10

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# ИНФОРМАЦИЯ
# ============================================================

print("=" * 60)
print("SOUL AI - TRAINING")
print("=" * 60)

print(f"Устройство: {DEVICE}")


# ============================================================
# ПРОВЕРКА DATASET
# ============================================================

if not os.path.exists(DATA_FILE):

    print()
    print("ОШИБКА: файл data.txt не найден.")
    print()
    print("Создай файл data.txt рядом с train.py")
    print("и добавь туда текст для обучения.")

    exit()


# ============================================================
# ЗАГРУЖАЕМ DATASET
# ============================================================

print()
print("Загрузка dataset...")

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print(
    f"Символов в dataset: {len(text):,}"
)


# ============================================================
# СОЗДАЁМ TOKENIZER
# ============================================================

print()
print("Создание tokenizer...")

tokenizer = SoulTokenizer()

tokenizer.train(
    [text],
    vocab_size=VOCAB_SIZE
)

print(
    f"Размер словаря: {len(tokenizer.vocab):,}"
)


# ============================================================
# СОХРАНЯЕМ TOKENIZER
# ============================================================

tokenizer.save(
    TOKENIZER_FILE
)

print(
    f"Tokenizer сохранён: {TOKENIZER_FILE}"
)


# ============================================================
# TOKENIZE DATASET
# ============================================================

print()
print("Tokenization...")

tokens = tokenizer.encode(
    text
)

tokens = torch.tensor(
    tokens,
    dtype=torch.long
)

print(
    f"Всего токенов: {len(tokens):,}"
)


# ============================================================
# ПРОВЕРКА РАЗМЕРА
# ============================================================

if len(tokens) <= BLOCK_SIZE:

    print(
        "ОШИБКА: dataset слишком маленький."
    )

    print(
        f"Нужно больше {BLOCK_SIZE} токенов."
    )

    exit()


# ============================================================
# CONFIG
# ============================================================

config = SoulConfig(

    vocab_size=len(tokenizer.vocab),

    block_size=BLOCK_SIZE,

    n_layers=N_LAYERS,

    n_heads=N_HEADS,

    d_model=D_MODEL,

    dropout=DROPOUT
)


# ============================================================
# MODEL
# ============================================================

print()
print("Создание Soul Transformer...")

model = SoulTransformer(
    config
)

model = model.to(
    DEVICE
)


# ============================================================
# PARAMETER COUNT
# ============================================================

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Параметров модели: {parameters:,}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 60)
print("НАЧАЛО ОБУЧЕНИЯ")
print("=" * 60)


model.train()


for epoch in range(EPOCHS):

    total_loss = 0.0

    steps = 0


    # Перемешиваем позиции
    max_start = len(tokens) - BLOCK_SIZE - 1


    # Количество шагов
    num_steps = max(
        1,
        max_start // BATCH_SIZE
    )


    for step in range(
        num_steps
    ):

        # ----------------------------------------
        # Создаём batch
        # ----------------------------------------

        batch_x = []
        batch_y = []


        for _ in range(
            BATCH_SIZE
        ):

            start = torch.randint(
                0,
                max_start,
                (
                    1,
                )
            ).item()


            x = tokens[
                start:
                start + BLOCK_SIZE
            ]


            y = tokens[
                start + 1:
                start + BLOCK_SIZE + 1
            ]


            batch_x.append(x)
            batch_y.append(y)


        x = torch.stack(
            batch_x
        ).to(DEVICE)


        y = torch.stack(
            batch_y
        ).to(DEVICE)


        # ----------------------------------------
        # Forward
        # ----------------------------------------

        logits, loss = model(
            x,
            y
        )


        # ----------------------------------------
        # Backpropagation
        # ----------------------------------------

        optimizer.zero_grad()

        loss.backward()


        # ----------------------------------------
        # Gradient clipping
        # ----------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )


        # ----------------------------------------
        # Update weights
        # ----------------------------------------

        optimizer.step()


        # ----------------------------------------
        # Statistics
        # ----------------------------------------

        total_loss += loss.item()

        steps += 1


        if step % 100 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Step {step}/{num_steps} | "
                f"Loss: {loss.item():.4f}"
            )


    # ========================================================
    # EPOCH RESULT
    # ========================================================

    average_loss = (
        total_loss / steps
    )


    print()
    print(
        f"Epoch {epoch + 1} завершён"
    )

    print(
        f"Средний Loss: {average_loss:.4f}"
    )

    print()


# ============================================================
# SAVE MODEL
# ============================================================

print("=" * 60)
print("СОХРАНЕНИЕ МОДЕЛИ")
print("=" * 60)


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "config": config.__dict__,

        "vocab_size":
            len(tokenizer.vocab)
    },

    MODEL_FILE
)


print(
    f"Модель сохранена: {MODEL_FILE}"
)


print()
print("=" * 60)
print("ОБУЧЕНИЕ SOUL ЗАВЕРШЕНО")
print("=" * 60)