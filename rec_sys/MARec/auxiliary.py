import torch.nn.functional as F
import torch.nn as nn
import jieba
import re


def bpr_loss(positive_predictions, negative_predictions, mask=None):
	loss = (1.0 - F.sigmoid(positive_predictions - negative_predictions))
	if mask is not None:
		mask = mask.float()
		loss = loss * mask
		return loss.sum() / mask.sum()
	return loss.mean()


class ScaledEmbedding(nn.Embedding):
	def reset_parameters(self):
		self.weight.data.normal_(0, 1.0 / self.embedding_dim)
		if self.padding_idx is not None:
			self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
	def reset_parameters(self):
		self.weight.data.zero_()
		if self.padding_idx is not None:
			self.weight.data[self.padding_idx].fill_(0)


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
