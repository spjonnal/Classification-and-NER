TASKS:
- Fine-tune a **transformer model** for **sentence classification**.
- Train an **NER model** for extracting key entities.

  Setup Instructions:

1) NumPy version 1.24.4
2) PyTorch version 2.1.0
3) Transformers version 4.37.2
4) Install PyTorch with CUDA support using the command -  !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


Training and Evaluation Guide for Classification and NER:

This project was developed and fine-tuned using Google Colab, which provides a free GPU environment.

- Upload the data input csv file or mount your google drive,
- Install above mentioned dependencies,
- Load data and split it for training and validation datasets while leaving out at least 10 samples from the corpus (for prediction),
- Tokenize the training and validation data,
- Fine tune the LLM on tokenized inputs and predict the output on the leftover 10 samples.

Evaluation Metrics:

- Calculate accuracy, F1, precision and recall scores.
output for Classification Task
accuracy of the classification =  1.0
F1 score =  1.0
precision score =  1.0
recall score =  1.0

Classification Prediction Example:

1) The text = the treatment resulted in full remission for the majority of patients., the predicted label of the text = Positive Outcome  , label's encoding = 2, and the original label = 2


2) The text = mild headaches were reported after the second dose of drugy., the predicted label of the text = Adverse Effect  , label's encoding = 0, and the original label = 0

Named Entity Recognition Example:

word tokens =  ['<s>', 'Pat', 'ients', 'Ġexperienced', 'Ġdizz', 'iness', 'Ġafter', 'Ġtaking', 'Ġ500', 'mg', 'Ġof', 'ĠDrug', 'A', '</s>']
The sentence = Patients experienced dizziness after taking 500mg of DrugA
Pat : O
ients : O
experienced : O
dizz : B-SYMPTOM
iness : B-SYMPTOM
after : O
taking : O
500 : B-DOSAGE
mg : O
of : O
Drug : O
A : O

Briefly outline how you would prepare the models for production (e.g., deploy as an API, optimize for speed) :

- By converting the model to an AP using Flask and Amazon Web Services can be used to deploy the API.

- I have implemented Lower Rank Adaptation technique which is useful in training only specific percentage of a model's training parameters (75% in my case) due to adapting a matrix multiplication of rank 8. However only 50% of the model's parameters were trained when adapted a rank of 4.

- The cost of time saving and lesser calculations is, lower prediction capabilities and to compensate, I had to increase the number of training epochs (without overfitting) which eventually took the same amount of time as when I trained the entire model's parameters with just ONE EPOCH while keeping the same learning rate.

- Thus, the choice of adapting LoRA depends on the type of the task performed (I would prefer for a less critical task like quick classification) and the model's performance.

Suggest ways to improve the models with more data (e.g., data augmentation, domain adaptation):

- We can use BioELECTRA model which is fine tuned on the abstracts from PubMed.
