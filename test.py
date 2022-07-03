import torch
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import copy
import warnings
warnings.filterwarnings("ignore")

def bert_top_keywords(model_name, text, reg_coeff=1, layers_to_reg=[], top_k=5):
  n_gram_range = (1, 1)
  count = CountVectorizer(ngram_range=n_gram_range, stop_words="english").fit([text])
  all_candidates = count.get_feature_names()
  nlp = spacy.load('en_core_web_sm')
  doc = nlp(text)
  noun_phrases = set(chunk.text.strip().lower() for chunk in doc.noun_chunks)
  nouns = set()
  for token in doc:
    if token.pos_ == "NOUN":
      nouns.add(token.text)
  all_nouns = nouns.union(noun_phrases)

  model = BertForSequenceClassification.from_pretrained("saved_models/checkpoint-211", num_labels=2, output_attentions=True, output_hidden_states=True)
  model.load_state_dict(torch.load("saved_models/bert-base-cased_Twitter_factual.pt", encoding="cp1252", map_location=device))
  tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
  candidates = list(filter(lambda candidate: candidate in all_nouns, all_candidates))
  candidate_tokens = tokenizer(candidates, padding=True, return_tensors="pt")
  candidate_embeddings = model.bert(**candidate_tokens)[1]
  text_tokens = tokenizer([text], padding=True, return_tensors="pt")
  text_embedding = model.bert(**text_tokens)[1]
  candidate_embeddings = candidate_embeddings.detach().numpy()
  text_embedding = text_embedding.detach().numpy()
  distances = cosine_similarity(text_embedding, candidate_embeddings)
  keywords = [candidates[index] for index in distances.argsort()[0][-top_k:]]
  print("\nBefore Regularization:", keywords)

  model_after_debiasing = copy.deepcopy(model)
  with torch.no_grad():
    for name, para in model_after_debiasing.named_parameters():
      if len(layers_to_reg) > 0:
        for layer in layers_to_reg:
          if str(layer) in name:
            para.data *= reg_coeff

  if reg_coeff != 1:
    candidate_embeddings_debiased = model_after_debiasing.bert(**candidate_tokens)[1]
    text_embedding_debiased = model_after_debiasing.bert(**text_tokens)[1]
    candidate_embeddings_debiased = candidate_embeddings_debiased.detach().numpy()
    text_embedding_debiased = text_embedding_debiased.detach().numpy()
    distances = cosine_similarity(text_embedding_debiased, candidate_embeddings_debiased)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_k:]]
    print("After Regularization:", keywords, "\n")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

text = "That little girl is noteable. It is elitist nazis like you that have ruined Wikipedia."
swapped_text = "That little boy is noteable. It is elitist nazis like you that have ruined Wikipedia."

bert_top_keywords("bert-base-cased", text, reg_coeff=0.1, layers_to_reg=[9], top_k=5)

bert_top_keywords("bert-base-cased", swapped_text, reg_coeff=0.1, layers_to_reg=[9], top_k=5)

# Results:
# Before Regularization: ['nazis', 'girl', 'wikipedia']
# After Regularization: ['wikipedia', 'girl', 'nazis'] 


# Before Regularization: ['boy', 'nazis', 'wikipedia']
# After Regularization: ['boy', 'wikipedia', 'nazis'] 