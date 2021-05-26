NumS2T

Math Word Problem Solving with Explicit Numerical Values

## Requirement

- Python3.6
- Pytorch 1.8.0
- numpy
- nltk

# Train the model.
python run_seq2tree.py

# Evaluate the model.
python evaluate.py

#Structure
├── README.md                   // help

├── data                        // datasets

│   ├── ape						// Ape210K dataset	

│   │   ├── train.ape.json 		//official data partition

│   │   ├── valid.ape.json

│   │   └── test.ape.json 

│   └── Math_23K.json           // Math23K dataset	

├── hownet						// external knowledge base HowNet

│   └── cilin.txt           	// external knowledge base cilin

├── models                      // Saved Models

├── output                      // Test data output

│ 

├── pre_data.py 				// data process

├── masked_cross_entropy.py		// cross_entropy function

├── expressions_transfer.py		// expression process

├── models.py					// NumS2T's main model structure

├── run_seq2tree.py				// train the model (Math23K default) 

├── run_seq2tree_APE.py			// train the model (APE210K default) 

└── evaluate.py 				// evaluate the model