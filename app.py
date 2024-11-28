import argparse
from ngrams import Ngram, corpus_from_file




def main(n, corpus, verbose):
    temperature = 1.0

    ngram = Ngram(n)
    ngram.fit(corpus, verbose=verbose)

    # Generar texto
    while True:
        # Selecciona que generar
        print_menu()
        option = input(color_text("\nOpción: ", "cyan"))
        context = ""
        if option not in ["4", "5"]:
            context = input(color_text("\nIngrese un contexto para generar texto: ", "cyan"))
        if option == "1":
            generated_text = ngram.generate(context, temperature)
        elif option == "2":
            generated_text = ngram.generate_scene(context, temperature)
        elif option == "3":
            generated_text = ngram.generate_episode(context, temperature)
        elif option == "4":
            temperature = float(input(color_text("\nIngrese la temperatura: ", "cyan")))
            continue
        elif option == "5":
            break
        else:
            print("Opción inválida")

        context += " " if context else ""
        print(color_text("\n" + context + generated_text, "purple"))

def print_menu():
    print(color_text("\n Seleccione una opción:", "green"))
    print(color_text("1. Generar linea de dialogo", "blue"))
    print(color_text("2. Generar escena", "blue"))
    print(color_text("3. Generar episodio", "blue"))
    print(color_text("4. Cambiar temperatura", "yellow"))
    print(color_text("5. Salir", "red"))

def color_text(text, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    return colors[color] + text + "\033[0m"

if __name__ == "__main__":
    # Crear un analizador de argumentos
    parser = argparse.ArgumentParser(
        description="Entrenar un modelo de n-gramas para generar texto basado en la serie Friends."
    )

    # Agregar argumentos
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Activar modo verbose para ver más detalles durante el entrenamiento",
    )

    parser.add_argument(
        "-n",
        type=int,
        required=True,
        help="Tamaño del n-grama",
        default=3
    )

    parser.add_argument(
        "-c",
        "--corpus",
        type=str,
        help="Ruta al archivo con el corpus de texto",
        default="datasets/seinfeld_corpus.txt"
    )

    # Parsear los argumentos
    args = parser.parse_args()

    corpus = corpus_from_file(args.corpus)

    # Llamar a la función principal con los argumentos proporcionados
    main(args.n, corpus, args.verbose)

