import argparse
import numpy as np

def load_weights_from_file(file_path):
    with open(file_path, 'r') as f:
        weights = [float(line.strip()) for line in f.readlines()]
    return np.array(weights)

def combine_weights(file1, file2, alpha=0.5, output_file=None):
    # If you don't pass me any file ..... I will put the merge result in the file 1
    if output_file is None:
        output_file = file1

    weights_1 = load_weights_from_file(file1)
    weights_2 = load_weights_from_file(file2)

    if len(weights_1) != len(weights_2):
        raise ValueError("I pesi dei due modelli devono avere la stessa lunghezza!")

    combined_weights = alpha * weights_1 + (1 - alpha) * weights_2

    with open(output_file, 'w') as f:
        for weight in combined_weights:
            f.write(f"{weight}\n")

    print(f"I pesi combinati sono stati salvati in '{output_file}'")

def main():
    parser = argparse.ArgumentParser(description="Combina i pesi di due modelli di machine learning.")
    
    parser.add_argument('file1', type=str, help="Il file contenente i pesi del primo modello")
    parser.add_argument('file2', type=str, help="Il file contenente i pesi del secondo modello")
    parser.add_argument('--alpha', type=float, default=0.5, help="Peso del primo modello (default: 0.5)")
    parser.add_argument('--output', type=str, default='combined_weights.txt', help="File di output per i pesi combinati (default: 'combined_weights.txt')")

    args = parser.parse_args()

    try:
        combine_weights(args.file1, args.file2, alpha=args.alpha, output_file=args.output)
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == '__main__':
    main()
