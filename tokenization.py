# the dataset is a single file to be able to run all the code in the repo on consumer devices
with open('the_verdict.txt', 'r') as file:
    raw_text = file.read()
print(f'Number of characters: {len(raw_text)}')

# steps: raw_text -> tokenization -> vocab -> token ids -> embedding

# tokenizing with a simple regex
import re
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text) # split by punctuation and whitespace
preprocessed = [item.strip() for item in preprocessed if item.strip() != ''] # remove empty strings
print(f'Number of tokens: {len(preprocessed)}')

# creating the vocabulary
all_tokens = sorted(list(set(preprocessed))) # remove duplicates and sort the tokens alphabetically
all_tokens.extend(['<|unk|>', '<|endoftext|>']) # add special tokens for unknown tokens and end of text
vocab_size = len(all_tokens)
print(f'Vocab size: {vocab_size}')
vocab = {token: i for i, token in enumerate(all_tokens)} # create a mapping from token to id. So vocab is a dictionary

# now we can define a simple tokenizer class
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: token for token, i in vocab.items()} # reverse the mapping
    
    def encode(self, text): # convert text to token ids
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text) # split by punctuation and whitespace
        preprocessed = [item.strip() for item in preprocessed if item.strip() != ''] # remove empty strings
        preprocessed = [token if token in self.str_to_int else '<|unk|>' for token in preprocessed] # replace unknown tokens with <|unk|>
        ids = [self.str_to_int[token] for token in preprocessed] # convert tokens to ids
        return ids
    
    def decode(self, ids):
        text = ' '.join([self.int_to_str[id] for id in ids]) # convert ids to tokens
        text = re.sub(r'\s+([,.?_!"()\'])', r'\1', text) # remove whitespace before punctuation
        return text

tokenizer = SimpleTokenizer(vocab)
text1 = "The fox jumps over the lazy dog"
text2 = "You are my sunshine, my only sunshine"
text3 = " <|endoftext|> ".join([text1, text2])
print(text3)
ids = tokenizer.encode(text3)
print(ids)
print(tokenizer.decode(ids))

#This tokenization strategy is very simple and not very effective. It doesn't handle unknown tokens well and doesn't take into account the structure of the text.
#We can improve this by using a more advanced tokenization strategy like byte pair encoding (BPE)


# now we will use byte pair encoding (BPE), which is a type of tokenization that is used by openAI's models
import tiktoken # written by tiktoken, it contains the BPE algorithm

gpt_tokenizer = tiktoken.get_encoding("o200k_base") # the encoding used for gpt-4o according to a sketchy Go documentation of tiktoken i found online

# print # of vocab
print(f'Vocab size of gpt-4o: {gpt_tokenizer.n_vocab}')

# usage similar to the tokenizer implemented above
ids = gpt_tokenizer.encode(text3, allowed_special={'<|endoftext|>'})

print(ids)
print(gpt_tokenizer.decode(ids))