import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, 
    Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

stop_words = set(stopwords.words('english'))

def setup_spark_environment():
    spark = (SparkSession.builder
        .appName("Advanced Document Classification")
        .config("spark.sql.shuffle.partitions", 200)
        .config("spark.executor.memory", "8g")
        .config("spark.driver.memory", "4g")
        .getOrCreate())
    return spark

def preprocess_text_udf(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def build_lstm_model(vocab_size, max_length, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, 
                  output_dim=128, 
                  input_length=max_length),
        Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)),
        Bidirectional(LSTM(64, dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    return model

def integrated_training_process(train_data, test_data):
    train_data['text'] = train_data['text'].apply(lambda x: preprocess_text_udf(x))
    test_data['text'] = test_data['text'].apply(lambda x: preprocess_text_udf(x))

    tokenizer = KerasTokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_data['text'])

    X_train = tokenizer.texts_to_sequences(train_data['text'])
    X_test = tokenizer.texts_to_sequences(test_data['text'])

    max_length = 150
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

    y_train = to_categorical(train_data['label'])
    y_test = to_categorical(test_data['label'])

    model = build_lstm_model(
        vocab_size=len(tokenizer.word_index),
        max_length=max_length,
        num_classes=len(train_data['label'].unique())
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        filepath='best_model.keras', 
        monitor='val_loss', 
        save_best_only=True
    )

    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=64, 
        validation_split=0.2,
        callbacks=[early_stop, model_checkpoint]
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    return model, history

def mapreduce_word_count(dataframe):
    word_count = (
        dataframe.rdd
        .flatMap(lambda row: [(word, 1) for word in row['text'].split()])
        .reduceByKey(lambda a, b: a + b)
        .toDF(['word', 'count'])
        .orderBy('count', ascending=False)
    )
    return word_count

def main():
    spark = setup_spark_environment()

    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    spark_train_df = spark.createDataFrame(train_data)

    preprocess_udf = udf(preprocess_text_udf, StringType())
    spark_train_df = spark_train_df.withColumn('cleaned_text', preprocess_udf(col('text')))

    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])
    model = pipeline.fit(spark_train_df)
    transformed_data = model.transform(spark_train_df)

    word_count = mapreduce_word_count(spark_train_df)
    word_count.show(10)

    model, history = integrated_training_process(train_data, test_data)

if __name__ == "__main__":
    main()
