# DLA_Project

Emotion Detection from Text using LSTM


1. Abstract
This project presents an Emotion Detection system that classifies textual data into six emotions: happy, sad, angry, fear, surprise, and calm. Using Long Short-Term Memory (LSTM) networks, the system captures sequential dependencies in text for accurate emotion classification. The dataset is synthetically generated with 300 samples (50 per emotion), preprocessed using standard NLP techniques, and modeled with a deep learning architecture. The model achieved promising results and can be applied in chatbots, mental health applications, and social media sentiment analysis.
________________________________________
2. Introduction
Emotion plays a vital role in human communication, and understanding emotions in text is essential for AI systems. Automatic emotion detection is widely used in:
•	Chatbots and virtual assistants
•	Social media sentiment analysis
•	Customer feedback and recommendation systems
•	Mental health monitoring
The project focuses on building a deep learning-based emotion classifier using LSTM networks, which are suitable for sequential data like text due to their ability to capture long-term dependencies.
________________________________________
3. Problem Statement
The goal is to:
1.	Preprocess raw text data
2.	Convert text into numerical representations suitable for deep learning
3.	Build an LSTM model to classify text into six predefined emotions
4.	Evaluate model performance using standard metrics
Challenges include:
•	Ambiguity in text emotions (some sentences can express multiple emotions)
•	Limited dataset size leading to possible overfitting
•	Variations in sentence structure, spelling, and slang
________________________________________
4. Tools and Technologies
Category	Tool/Library	Purpose
Programming Language	Python	Development
Data Handling	Pandas, NumPy	Dataset creation and preprocessing
NLP	NLTK	Tokenization, stopwords removal
Deep Learning	TensorFlow/Keras	LSTM model implementation
Preprocessing	sklearn	Label encoding, train-test split
Visualization	matplotlib, seaborn	Plots and charts
________________________________________
5. Dataset Description
•	Dataset Size: 300 samples (50 per emotion)
•	Emotions: happy, sad, angry, fear, surprise, calm
•	Format: CSV file with columns text and emotion
Sample Dataset:
Text	Emotion
I am feeling very happy today!	happy
I feel so lonely and down.	sad
I am furious right now!	angry
I’m scared of what might happen next.	fear
Wow, I didn’t expect that!	surprise
I feel peaceful and relaxed.	calm
Visualization:
•	Class Distribution: All six emotions are balanced in the dataset.
•	Word Frequency: Most common words indicate typical emotion context (e.g., “happy”, “excited”, “joy” for happy).
________________________________________
6. Data Preprocessing
1.	Text Cleaning:
o	Convert text to lowercase
o	Remove punctuation, numbers, and special characters
o	Remove stopwords
o	Tokenize text into words
2.	Text Representation:
o	Use Tokenizer to convert words into sequences of integers
o	Pad sequences to ensure uniform length (max length = 50)
3.	Label Encoding:
o	Convert emotion labels into numeric format using LabelEncoder
4.	Train-Test Split:
o	80% training data, 20% testing data
________________________________________
7. Model Architecture
The LSTM model architecture is as follows:
1.	Embedding Layer: Converts words into dense 64-dimensional vectors
2.	LSTM Layer: 128 units, captures sequential dependencies
3.	Dropout Layer: 30% dropout to reduce overfitting
4.	Dense Layer: 64 neurons with ReLU activation
5.	Dropout Layer: 30%
6.	Output Layer: Softmax activation for 6-class classification
Hyperparameters:
•	Loss Function: sparse categorical crossentropy
•	Optimizer: Adam
•	Batch Size: 32
•	Epochs: 10
________________________________________
8. Model Training
•	Model trained for 10 epochs
•	Training accuracy gradually increases
•	Validation accuracy remains close to training accuracy, indicating no overfitting
•	Visualization: Accuracy vs. Epochs plot shows consistent learning
________________________________________
11. Model Evaluation
Metric	Value
Test Accuracy	~90% (depends on random initialization)
Confusion Matrix	Shows correct and incorrect classifications per emotion
•	Precision, Recall, and F1-Score were calculated for each class.
•	Misclassifications mostly occur between emotions with similar contexts (e.g., fear vs. surprise).
Sample Confusion Matrix:
	happy	sad	angry	fear	surprise	calm
happy	9	0	0	0	0	1
sad	0	8	1	1	0	0
angry	0	0	9	0	1	0
fear	0	0	0	9	1	0
surprise	0	0	1	0	9	0
calm	0	0	0	0	0	10
________________________________________


12. Sample Predictions
Input Text	Predicted Emotion
I am feeling awesome today!	happy
I am so scared right now.	fear
I am really frustrated.	angry
Life feels simple and serene.	calm
________________________________________
13. Challenges
•	Small synthetic dataset may not represent real-world text diversity
•	Short text length can make emotion detection harder
•	Ambiguity between emotions like fear and surprise
________________________________________
14. Future Work
•	Use real-world datasets like social media posts or customer feedback
•	Incorporate pre-trained embeddings (GloVe, Word2Vec, BERT) for better performance
•	Expand emotion categories to include more nuanced emotions
•	Deploy as a web application or chatbot
•	Combine multi-modal data (text + audio + video) for advanced emotion detection
________________________________________
15. Conclusion
This project successfully demonstrates emotion detection from text using LSTM. The model achieved promising accuracy using a small dataset. With enhancements such as larger datasets and pre-trained embeddings, the model can be scaled for real-world applications like sentiment analysis, chatbots, and mental health monitoring.
