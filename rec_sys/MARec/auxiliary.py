import torch.nn as nn
import jieba
import re
from settings import *
from torch import save


class KEmbedding(nn.Embedding):
	def __init__(self, num_embeddings, embedding_dim, k):
		self.k = k
		super().__init__(num_embeddings, embedding_dim)

	def reset_parameters(self) -> None:
		self.weight.data += self.k


class UniformEmbedding(nn.Embedding):
	def __init__(self, num_embeddings, embedding_dim, low, high):
		self.low = low
		self.high = high
		super().__init__(num_embeddings, embedding_dim)

	def reset_parameters(self) -> None:
		self.weight.data.uniform_(self.low, self.high)


class EarlyStopping:

	def __init__(self, patience=5, delta=0.005, path=prefix+"model.pickle", trace_fnc=print, verbose=True):
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.delta = delta
		self.path = path
		self.trace_func = trace_fnc
		self.save_model = None

	def __call__(self, score, model):
		if self.best_score is None or score > self.best_score - self.delta:
			if self.best_score is None:
				self.best_score = score
			else:
				self.best_score = max(self.best_score, score)
			self.save_model = model
			self.counter = 0
		else:
			self.counter += 1
			if self.counter >= self.patience:
				save(model, self.path)
				self.trace_func("Early Stop Trigger, best NDCG: %.4f" % self.best_score)
				return True

		return False


class CutWord:
	def __init__(self, cut_all=False):
		self.cut_all = cut_all
		self.stop_words = self.load_stop_word()
		jieba.initialize()
		jieba.load_userdict('./material/mydict.txt')
		self.pattern = [
			[re.compile(r'&\w+;'), 'emo'],
			[re.compile(r'{:soso\w+:}'), 'soso'],
			[re.compile(r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'), 'url']
		]
		self._word = re.compile("[\u4e00-\u9fa5A-Za-z]+")

	@staticmethod
	def load_stop_word():
		with open('./material/stopwords.txt', 'r', encoding='utf-8') as f:
			lines = f.readlines()
			stop_words = set()
			for i in lines:
				i = i.replace('\n', "")
				stop_words.add(i)

		return stop_words

	def cut(self, content: str):
		for p, w in self.pattern:
			content = p.sub(w, content)
		content = self._word.findall(content)
		new_text = " ".join(content)
		seg_list_exact = jieba.cut(new_text, cut_all=self.cut_all)
		result_list = []

		for word in seg_list_exact:
			if len(word) > 1 and word not in self.stop_words:
				result_list.append(word)
		return result_list
