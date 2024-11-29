import argparse
from ngrams import Ngram, corpus_from_file

def main(n, file_path, verbose):
    try:
        corpus = corpus_from_file(file_path)
        temperature = 1.0
        ngram = Ngram(n)
        ngram.fit(corpus, verbose=verbose)

        while True:
            option = get_user_option(temperature)
            if option in {"0", "1", "2", "3"}:
                generated_text = generate_text(option, ngram, temperature)
                print(color_text("\n" + " " + generated_text, "purple"))
            elif option == "4":
                try:
                    temperature = float(input(color_text("\nIngrese la temperatura: ", "cyan")))
                except ValueError:
                    print(color_text("La temperatura debe ser un número válido.", "red"))
            elif option == "5":
                break
            else:
                print("Opción inválida")

    except KeyboardInterrupt:
        print("\nPrograma interrumpido. Saliendo...")

def get_user_option(current_temperature):
    print_menu(current_temperature)
    return input(color_text("\nOpción: ", "cyan"))

def generate_text(option, ngram, temperature):
    if option == "0":
        print(color_text("Mencionar nombres de personajes o títulos de episodios, dado que el corpus es reducido.", "yellow"))
        context = input(color_text("\nIngrese el contexto: ", "cyan"))
        return ngram.generate_scene(context, temperature)
    elif option == "1":
        return ngram.generate_line(temperature=temperature)
    elif option == "2":
        return ngram.generate_scene(temperature=temperature)
    elif option == "3":
        return ngram.generate_episode(temperature=temperature)
    else:
        print("Opción inválida")

def print_menu(current_temperature):
    options = [
        ("0. Generar escena con contexto", "blue"),
        ("1. Generar linea de dialogo", "blue"),
        ("2. Generar escena", "blue"),
        ("3. Generar episodio", "blue"),
        (f"4. Cambiar temperatura ({current_temperature:.2f})", "yellow"),
        ("5. Salir", "red")
    ]
    print(color_text("\n Seleccione una opción:", "green"))
    for text, color in options:
        print(color_text(text, color))

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
    parser = argparse.ArgumentParser(
        description="Entrenar un modelo de n-gramas para generar texto basado en la serie Friends."
    )

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
        help="Tamaño del n-grama"
    )

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        help="Ruta al archivo con el corpus de texto",
        default="datasets/friends_corpus.txt"
    )

    args = parser.parse_args()
    main(args.n, args.file_path, args.verbose)
