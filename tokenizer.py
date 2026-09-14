import re
import json
from collections import Counter


class SoulTokenizer:
    """
    Простой токенизатор Soul AI.

    Он разбивает текст на:
    - слова
    - числа
    - знаки препинания
    - специальные токены
    """

    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ]

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

        self.vocab_size = 0

    # ==========================================
    # TOKENIZE
    # ==========================================

    def tokenize(self, text):
        """
        Превращает текст в список токенов.
        """

        text = text.lower().strip()

        tokens = re.findall(
            r"\w+|[^\w\s]",
            text,
            re.UNICODE
        )

        return tokens

    # ==========================================
    # BUILD VOCABULARY
    # ==========================================

    def train(self, texts, vocab_size=5000):
        """
        Создаёт словарь на основе нашего датасета.
        """

        counter = Counter()

        for text in texts:

            tokens = self.tokenize(text)

            counter.update(tokens)

        # Добавляем специальные токены

        vocabulary = list(
            self.SPECIAL_TOKENS
        )

        # Самые частые токены

        most_common = counter.most_common(
            vocab_size - len(vocabulary)
        )

        for token, _ in most_common:

            if token not in vocabulary:

                vocabulary.append(token)

        # Создаём словари

        self.token_to_id = {
            token: index
            for index, token
            in enumerate(vocabulary)
        }

        self.id_to_token = {
            index: token
            for token, index
            in self.token_to_id.items()
        }

        self.vocab_size = len(
            self.token_to_id
        )

        print(
            f"Soul Tokenizer создан."
        )

        print(
            f"Размер словаря: {self.vocab_size}"
        )

    # ==========================================
    # ENCODE
    # ==========================================

    def encode(
        self,
        text,
        add_bos=True,
        add_eos=True
    ):
        """
        Превращает текст в ID токенов.
        """

        tokens = self.tokenize(text)

        result = []

        if add_bos:

            result.append(
                self.token_to_id["<BOS>"]
            )

        for token in tokens:

            token_id = self.token_to_id.get(
                token,
                self.token_to_id["<UNK>"]
            )

            result.append(token_id)

        if add_eos:

            result.append(
                self.token_to_id["<EOS>"]
            )

        return result

    # ==========================================
    # DECODE
    # ==========================================

    def decode(self, ids):
        """
        Превращает ID обратно в текст.
        """

        tokens = []

        for token_id in ids:

            token = self.id_to_token.get(
                token_id,
                "<UNK>"
            )

            if token in self.SPECIAL_TOKENS:
                continue

            tokens.append(token)

        text = ""

        for token in tokens:

            # Знаки препинания
            # не должны иметь пробел перед собой

            if re.match(
                r"[^\w\s]",
                token,
                re.UNICODE
            ):

                text += token

            else:

                if text:
                    text += " "

                text += token

        return text

    # ==========================================
    # SAVE
    # ==========================================

    def save(self, path):

        data = {
            "token_to_id":
                self.token_to_id,

            "id_to_token":
                {
                    str(k): v
                    for k, v
                    in self.id_to_token.items()
                },

            "vocab_size":
                self.vocab_size
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"Tokenizer сохранён: {path}"
        )

    # ==========================================
    # LOAD
    # ==========================================

    def load(self, path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        self.token_to_id = \
            data["token_to_id"]

        self.id_to_token = {
            int(k): v
            for k, v
            in data["id_to_token"].items()
        }

        self.vocab_size = \
            data["vocab_size"]

        print(
            f"Tokenizer загружен: {path}"
        )


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    dataset = [
        "Привет! Меня зовут Soul.",
        "Я искусственный интеллект.",
        "Я хочу научиться разговаривать.",
        "Моя цель — помогать человеку.",
        "Мы создаём Soul AI вместе.",
        "Это только начало."
    ]

    tokenizer = SoulTokenizer()

    tokenizer.train(
        dataset,
        vocab_size=100
    )

    text = "Привет! Я Soul."

    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode(encoded)

    print()
    print("Исходный текст:")
    print(text)

    print()
    print("Токены:")
    print(tokenizer.tokenize(text))

    print()
    print("ID:")
    print(encoded)

    print()
    print("Обратно:")
    print(decoded)

    tokenizer.save(
        "soul_tokenizer.json"
    )