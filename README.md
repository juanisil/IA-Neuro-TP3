<!-- Ignore  -->

# IA-Neuro-TP3

## Entrenamiento de Modelo de N-Gramas

El script CLI.py entrena un modelo de **n-gramas** para generar texto basado en un corpus, por defecto utilizando la serie *Friends*.

### Uso

```bash
python CLI.py -v
```

### Doc

```bash
python CLI.py [-n <tamaño_ngramas>] [-c <ruta_corpus>] [-v]
```

### Parámetros

<!-- create table -->
| Parámetro | Descripción            | Valor por defecto             |
| --------- | ---------------------- | ----------------------------- |
| `-n`      | Tamaño de los n-gramas | `5`                           |
| `-c`      | Ruta al corpus         | `datasets/friends_corpus.txt` |
| `-v`      | Modo verbose           | `False`                       |