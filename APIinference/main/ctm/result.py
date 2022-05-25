from typing import List


class Result:
    class Word:
        def __init__(self, start: float, duration: float, word: str, conf: float):
            self.start = start
            self.duration = duration
            self.word = word
            self.conf = conf

        def json(self):
            return {
                'start': round(self.start, 2),
                'duration': round(self.duration, 2),
                'word': self.word,
                'conf': round(self.conf, 2),
            }

        def json_end(self):
            return {
                'start': round(self.start, 2),
                'end': round(self.start + self.duration, 2),
                'word': self.word,
                'conf': round(self.conf, 2),
            }

    def __init__(self):
        self.__l: List[Result.Word] = []

    def append(self, start: float, duration: float, word: str, conf: float = -1):
        self.__l.append(Result.Word(start, duration, word, conf))

    def __len__(self):
        return len(self.__l)

    def update_words(self, wrd_list: List[str]):
        if len(wrd_list) != len(self):
            raise ValueError('Different lengths')
        for i, wrd in enumerate(wrd_list):
            self.__l[i].word = wrd

    def json(self) -> list:
        return [w.json() for w in self.__l]

    def json_end(self) -> list:
        return [w.json_end() for w in self.__l]

    def transcription(self) -> str:
        return ' '.join(w.word for w in self.__l)

    def to_sils(self) -> str:
        out = []
        prev = 0
        for i, wrd in enumerate(self.__l):
            if i > 0:
                out.append(f'<sil={wrd.start-prev:.3f}>')
            out.append(wrd.word)
            prev = wrd.start + wrd.duration
        return ' '.join(out)
