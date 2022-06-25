import os
from alive_progress import alive_bar


def main():
    with open("exceptions.txt", "r") as file:
        lines = list([line for line in file])

        with alive_bar(len(lines), title="Line") as bar:
            for word in lines:
                print(word)
                os.system('pause')
                os.system('cls' if os.name == 'nt' else 'clear')

                bar()


if __name__ == '__main__':
    main()
