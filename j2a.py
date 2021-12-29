import csv
import sys
import pykakasi


def shogi_replace(moji):
    r = moji
    r = r.replace("歩", "fu")
    r = r.replace("香車", "kyousha").replace("香", "kyou")
    r = r.replace("桂馬", "keima").replace("桂", "kei")
    r = r.replace("銀", "gin")
    r = r.replace("金", "kin")
    r = r.replace("飛車", "hisha").replace("飛", "hi")
    r = r.replace("角", "kaku")
    r = r.replace("玉", "gyoku")
    r = r.replace("同", "dou")
    r = r.replace("1", "ichi")
    r = r.replace("2", "ni")
    r = r.replace("3", "san")
    r = r.replace("4", "yon")
    r = r.replace("5", "go")
    r = r.replace("6", "roku")
    r = r.replace("7", "nana")
    r = r.replace("8", "hachi")
    r = r.replace("9", "kyu")
    r = r.replace("相", "ai")
    r = r.replace("１", "ichi")
    r = r.replace("２", "ni")
    r = r.replace("３", "san")
    r = r.replace("４", "yon")
    r = r.replace("５", "go")
    r = r.replace("６", "roku")
    r = r.replace("７", "nana")
    r = r.replace("８", "hachi")
    r = r.replace("９", "kyu")
    r = r.replace("、", " ")
    r = r.replace(",", " ")
    return r


def j2a(txt):
    txt_list = []
    with open(txt, encoding="utf-8", newline="") as f:
        for cols in csv.reader(f, delimiter="\t"):
            kakasi = pykakasi.kakasi()
            kakasi.setMode("H", "a")
            kakasi.setMode("K", "a")
            kakasi.setMode("J", "a")
            conv = kakasi.getConverter()
            r = shogi_replace(cols[8])
            txt_list.append([cols[2], cols[4], conv.do(r)])
    return txt_list
