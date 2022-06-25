import io
import re
import os
import glob
import argparse
import requests
from train import clean
from bs4 import BeautifulSoup
from alive_progress import alive_bar
from langdetect import detect, LangDetectException


parser = argparse.ArgumentParser()
parser.add_argument("--erase", help="erase existing scripts", action="store_true")
args = parser.parse_args()


def scripts():
    if args.erase:
        for file in glob.iglob("scripts/*.txt"):
            os.remove(file)

    genres = re.sub("[<].*?[>]", "", str(BeautifulSoup(
        requests.get(
            "https://imsdb.com/"
        ).content,
        "html.parser"
    ).find_all("table")[4])).strip().split()[1:]

    with io.open("conflicts.txt", "w", encoding="utf-8") as log:
        for genre in genres:
            movies = BeautifulSoup(
                requests.get(
                    "https://imsdb.com/genre/" + genre
                ).content,
                "html.parser"
            ).find_all("p")

            with alive_bar(len(movies), title=genre) as bar:
                for movie in movies:
                    title = movie.a.contents[0]
                    file = "scripts/" + re.sub("[^a-z\\d_]", "", re.sub("\\s+", "_", title.lower())) + ".txt"

                    if not os.path.isfile(file):
                        try:
                            script = requests.get(
                                "https://imsdb.com" + BeautifulSoup(
                                    requests.get(
                                        "https://imsdb.com" + movie.a["href"]
                                    ).content,
                                    "html.parser"
                                ).find_all("p")[-1].a["href"]
                            ).text

                            if detect(script) != "en":
                                continue

                            script = script[script.index('<td class="scrtext">') + 20:]
                            script = script[:script.index("</pre>")]
                            script = clean(script)
                        except (TypeError, IndexError, ValueError, LangDetectException) as e:
                            continue

                        with open(file, "w+") as txt:
                            txt.write(script)

                    bar()


if __name__ == '__main__':
    scripts()
