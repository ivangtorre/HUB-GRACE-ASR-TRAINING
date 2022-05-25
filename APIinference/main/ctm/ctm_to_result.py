from .result import Result


def parse_line(line):
    l = line.strip().split()
    if len(l) == 5:
        return l
    if len(l) == 4:
        return [None, *l]
    if len(l) == 6:
        return [None, l[0], l[1], l[4], l[5]]
    return [None]*5


def ctm2result(ctm_string: str) -> Result:
    r = Result()
    for line in ctm_string.split('\n'):
        if not line.strip():
            continue
        audio, start, duration, word, conf = parse_line(line.strip())
        r.append(
            start=float(start),
            duration=float(duration),
            word=word,
            conf=float(conf)
        )
    return r
