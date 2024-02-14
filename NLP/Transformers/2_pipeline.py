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

# pipeline'lar ile
"""
available tasks are ['audio-classification', 'automatic-speech-recognition', 'conversational', 'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask', 'image-classification', 'image-segmentation',
'image-to-text', 'ner', 'object-detection', 'question-answering', 'sentiment-analysis', 'summarization',
'table-question-answering', 'text-classification', 'text-generation', 'text2text-generation', 'token-classification', 'translation', 'visual-question-answering', 'vqa', 'zero-shot-classification', 'zero-shot-image-classification',
'zero-shot-object-detection', 'translation_XX_to_YY']"
"""

import pandas as pd
from transformers import pipeline

# #######################################################################
# Sentiment Analysis

classifier = pipeline("sentiment-analysis")

text = "It's great to learn NLP for me."

outputs = classifier(text)
print(outputs)

outputs = pd.DataFrame(outputs)
print(outputs)
print('Sentiment Analysis Done.')


# ########################################################################
# Zero-Shot Classifier

classifier_2 = pipeline("zero-shot-classification")

labels = ['education', 'business', 'tech']
text = "This is tutorial about Hugging Face"

outputs = classifier_2(text, classifier)

outputs = pd.DataFrame(outputs)
print(outputs)
print('Zero-Shot Classification Done.')


# ########################################################################
# Image Sengmentation

classifier_3 = pipeline("image-segmentation")


image = '' # image path
outputs = classifier_3(image)

outputs = pd.DataFrame(outputs)
print(outputs)
print('Image Segmentation Done.')


# ########################################################################
# Img2Text

classifier_4 = pipeline("image-to-text")

image = '' # image path
outputs = classifier_4(image)

print(outputs)
print('Img2Text Done.')


# ########################################################################
# Img2Img

classifier_5 = pipeline("image-to-image")

image = '' # image path
outputs = classifier_5(image)

print(outputs)
print('Img2Img Done.')


# ########################################################################
# Text2Speech

classifier_6 = pipeline("text-to-speech")

text = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry.'

outputs = classifier_6(text)

import scipy.io.wavfile as wavf

output_name = 'output.wav'
samples = outputs['audio'][0]
sampling_rate = outputs['sampling_rate']

wavf.write(output_name, sampling_rate, samples)

print('Text2Speech Done.')



print('Done..')