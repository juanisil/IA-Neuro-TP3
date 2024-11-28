# IA-Neuro-TP3

## Entrenamiento de Modelo de N-Gramas

Este script entrena un modelo de **n-gramas** para generar texto basado en un corpus, por defecto utilizando la serie *Seinfeld*.

### Uso

```bash
python app.py -n <tamaño_ngramas> [-c <ruta_corpus>] [-v]
```

### Parámetros

<!-- create table -->
| Parámetro | Descripción            | Valor por defecto              |
| --------- | ---------------------- | ------------------------------ |
| `-n`      | Tamaño de los n-gramas | _Requerido_                    |
| `-c`      | Ruta al corpus         | `datasets/seinfeld_corpus.txt` |
| `-v`      | Modo verbose           | `False`                        |

### Ejemplo

```bash
python app.py -n 3 -c datasets/friends_corpus.txt -v
```
