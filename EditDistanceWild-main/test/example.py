import sys
import torch
from torch.nn.utils.rnn import pad_sequence
from EditDistanceWild import editdistance, editdistance1N

padToken = 0

def encode(seq: str):
	token_seq = [ord(c) for c in seq]
	return torch.tensor(token_seq, dtype=torch.long)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('WORKING on',device)

ref = ['hello','test','test','TESTt','?bcd','a?cd','abc?','??cd','*bcd','a*d','abc*','*bcd','*bcdef']
hyp = ['Hellf','tst','etest','test','abcd','abcd','abcd','a?cd','xxabcd','abxxcd','abcdxx','xxhdgfhbcd','abc']

# ENCODE CHARS AS INTEGERS
ref = [encode(w) for w in ref]
hyp = [encode(w) for w in hyp]

# PADDING
x = pad_sequence(ref, batch_first=True, padding_value=padToken)
y = pad_sequence(hyp, batch_first=True, padding_value=padToken)

# MOVE TO DEVICE (if any)
x, y = x.to(device), y.to(device)

# COMPUTE EDITDISTANCE WITH WILDCARDS BETWEEN TWO ARRAYS (weights ins/del=1, sub=2)
pred = editdistance(x, y, padToken, 1, 2).to('cpu')
print(pred)

ref1 = ['hello']
ref1 = [encode(w) for w in ref1]
x1 = pad_sequence(ref1, batch_first=True, padding_value=padToken)
x1 = x1.to(device)

# COMPUTE EDITDISTANCE WITH WILDCARDS BETWEEN ONE WORD AND ONE ARRAY
pred = editdistance1N(x1, y, padToken, 1, 2).to('cpu')
print(pred)
