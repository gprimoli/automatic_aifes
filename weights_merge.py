import argparse
import numpy as np
import logging
import sys

# Configurazione logger
logger = logging.getLogger("WeightCombiner")
logger.setLevel(logging.DEBUG)  # Cambia a INFO o WARNING per meno verbosità

# Handler per la console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def load_weights_from_file(file_path):
    logger.debug(f"Caricamento pesi da '{file_path}'")
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            weights = [np.array([float(val) for val in line.strip().split(',')]) for line in lines if line.strip()]
        logger.info(f"{len(weights)} tensori caricati da '{file_path}'")
        return weights  # Lista di array (uno per riga)
    except Exception as e:
        logger.error(f"Errore durante il caricamento dei pesi da '{file_path}': {e}")
        raise

def combine_weights(file1, file2, alpha=0.5, output_file=None):
    if output_file is None:
        output_file = file1
        logger.info(f"Nessun file di output specificato. I pesi combinati verranno salvati in '{file1}'")

    logger.info(f"Combinazione dei pesi con alpha = {alpha}")
    weights_1 = load_weights_from_file(file1)
    weights_2 = load_weights_from_file(file2)

    if len(weights_1) != len(weights_2):
        logger.error("I modelli hanno un numero diverso di tensori!")
        raise ValueError("I due modelli devono avere lo stesso numero di tensori")

    combined = []
    for idx, (w1, w2) in enumerate(zip(weights_1, weights_2)):
        if len(w1) != len(w2):
            raise ValueError(f"Tensore {idx} ha lunghezze diverse tra i due modelli!")
        c = alpha * w1 + (1 - alpha) * w2
        combined.append(c)

    try:
        with open(output_file, 'w') as f:
            for tensor in combined:
                line = ','.join(f"{val:.7f}" for val in tensor)
                f.write(line + '\n')
        logger.info(f"I pesi combinati sono stati salvati in '{output_file}'")
    except Exception as e:
        logger.error(f"Errore durante il salvataggio dei pesi in '{output_file}': {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Combina i pesi di due modelli di machine learning.")
    print("Qua sono entrato");
    parser.add_argument('file1', type=str, help="Il file contenente i pesi del primo modello")
    parser.add_argument('file2', type=str, help="Il file contenente i pesi del secondo modello")
    parser.add_argument('--alpha', type=float, default=0.5, help="Peso del primo modello (default: 0.5)")
    parser.add_argument('--output', type=str, default="toload.csv", help="File di output per i pesi combinati")

    args = parser.parse_args()

    try:
        combine_weights(args.file1, args.file2, alpha=args.alpha, output_file=args.output)
    except Exception as e:
        logger.exception("Errore durante l'esecuzione dello script")

if __name__ == '__main__':
    main()
