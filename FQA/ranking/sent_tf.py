# from sentence_transformers import SentenceTransformer, models

# word_embedding_model = models.Transformer('/home/jie/Documents/JIE_NLP/NLPProjects/FQA/lib/bert/', max_seq_length=256)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

# model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

# #Define your train examples. You need more than just two examples...
# train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#     InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

# #Define your train dataset, the dataloader and the train loss
# train_dataset = SentencesDataset(train_examples, model)
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
# train_loss = losses.CosineSimilarityLoss(model)

# #Tune the model
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
from transformers import pipeline
nlp = pipeline('fill-mask')
nlp("")