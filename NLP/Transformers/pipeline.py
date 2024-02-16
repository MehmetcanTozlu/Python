# Pipeline'lar ile NLP Giris
# NLP(Natural Language Processing), metin ve ses verilerini analiz etmeyi amaclar.

# NLP alaninda son zamanlarda, Hugging Face tarafindan "Transformers" kutuphanesi gelistirilmistir.
# Transformers kutuphanesi ile,
# - Text Classification,
# - Text Summarization,
# - Text Generation,
# - Question Answering
# - Entity Recognition
# - Translation
# - Zero-shot Classification
# cok basit bir sekilde yapilabilir.

from transformers import pipeline
import pandas as pd


# Text Generation

classifier_text_generate = pipeline('text-generation')

prompt = 'This tutorial will walk you through how to'

outputs = classifier_text_generate(prompt, max_length=100)
print(outputs[0]['generated_text'])


classifier_text_generate2 = pipeline('text-generation', model='distilgpt2')

outputs = classifier_text_generate2(prompt, max_length=50)
print(outputs[0]['generated_text'])



# Query Answer

classifier_qa = pipeline('question-answering')

text = 'My name is Can. I love Los Angeles'
quenstion = 'Where do I like?'

outputs = classifier_qa(question=quenstion, context=text)
print(pd.DataFrame([outputs]))



# Translation

classifier_translation = pipeline('translation_en_to_de')

text = 'I hope you enjoy it'

outputs = classifier_translation(text, clean_up_tokenization_spaces=True)
print(outputs, '\n', outputs[0]['translation_text'])