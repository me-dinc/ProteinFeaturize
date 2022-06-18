Usage:

python main.py <input file> <format> <size> <encodings>

input file: A fasta file containing amino acid sequences
format: "dict" -> Result is saved as a pickle file. Inside the file, there is a dictionary where protein names are keys and feature matrices(list of numpy arrays) are values.
	"seperate"  -> For each type of encoding, a seperate folder is created. Feature matrices of each protein are saved inside respective folders as seperate files. 
size: (int) size of feature matrices
encodings: Desired encodings, seperated by a single space. 

Example:

python main.py example.fasta dict 50 2DEncodings blosum62 GRAR740104 SIMK990101 ZHAC000103
