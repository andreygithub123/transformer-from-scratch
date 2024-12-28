import torch
import torch.nn as nn
import math 
import numpy


class InputEmbedings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

# inputEmb= InputEmbedings(d_model=5,vocab_size=10)
# listA = torch.randint(low=0,high=3,size=(2,2))
# print(listA)
# print(inputEmb.forward(listA))

class PositionalEncoding(nn.Module):

    def __init__(self,d_model: int , seq_length:int, dropout: float) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_length=seq_length
        self.dropout=nn.Dropout(dropout) # prevents overfitting by setting 0 some random parameters and scaling the rest

        # create a positional encoding matrix of shape(seq_length, d_model)
        pe=torch.zeros(seq_length,d_model)
        position = torch.arange(0,seq_length,dtype=torch.float).unsqueeze(1) # position vector to store the positions of tokens
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        # apply sin to even positions and cosine to odd positions
        # [start:stop:step]
        pe[:,::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1,seq_len,d_model)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad(False)
        return self.dropout(x)

# pe=torch.zeros(5,4)
# print("Initial matrix\n",pe)
# print("------------------------------")
# even_number = 1
# odd_number = 3
# pe[::2] += even_number
# pe[1::2] += odd_number
# print("Matrix after opertaion",pe)

# listA = torch.arange(0,5,dtype=torch.float).unsqueeze(-1)
# listB = torch.arange(0,5,dtype=torch.float).unsqueeze(-2)
# print(listA,listA.shape)
# print(listB,listB.shape)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6)->None:
        super().__init__()
        #epsilon is used for numerical stability ( to big numbers ) and mitigate divison by 0 problem
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1)) # Scale - Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Shit - Added

    def forward(self,x):
        mean = x.mean(dim =-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha*( x-mean ) / (std+self.eps) + self.bias
    


# x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2, 3)

# # Compute the mean along the last dimension (columns)
# # mean is seen as a dimensionality reduction prodecure ( keepdim=True  keeps the original shape)
# # dim = 0 => reduces that dimensions; if dim=0 = rows => rows are reduced so it;s the mean along columns
# mean = x.mean(dim=1,keepdim=True)  # Default keepdim=False
# print(mean)  # Output: tensor([2.0000, 5.0000])
# print(mean.shape)  # Output: torch.Size([2])

class FeedForwardBlock(nn.Module):

    def __init__(self,d_model: int , d_ff : int , dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) # W2 and B2 


    def forward(self,x):
        # ( Batch, Seq_len, d_model) --> (Batch,Seq_Len,d_ff) --> ( Batch, Seq_Len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model:int, h_heads:int, dropout: float) ->None:
        super().__init__()
        self.d_model = d_model
        self.h_heads = h_heads
        assert d_model % h_heads ==0, "d_model is not divisible by h_heads"

        self.d_k = d_model // h_heads # rounded division ( keeps only the integer )=> convert to integer type
        self.w_q = nn.Linear(d_model,d_model,bias=False) # Wq
        self.w_k = nn.Linear(d_model,d_model,bias=False) # Wk
        self.w_v = nn.Linear(d_model,d_model,bias=False) # Wv

        self.w_o = nn.Linear(d_model,d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    # NOTE : static methods belongs to the class not to an instance of the class => can be called in other circumstances
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]    

        # ( Batch, h, Seq_len, d_k) --> (Batch , h, Seq_Len, Seq_Len)
        attention_scores = ( query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0 ,-1e9 )
        attention_scores = attention_scores.softmax(dim = -1) # ( Batch, h ,seq_len, seq_len )
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return ( attention_scores@ value), attention_scores

    def forward(self,q,k,v, mask):
        # mask => if we want some words not interact with other words
        # NOTE : self.w_q = nn.Linear => initializing a randpm learnable matrix Wq ( d_model,d_model) but no operation is being made
        # when applying query = self.w_q(q) we make the operation :query = q x Wq(T) , bias disabed => Linear Transformation
        query = self.w_q(q)  # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        key=self.w_k(k) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        value=self.w_v(v) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)

        # divide q',k',v' into smaller heads
        # NOTE : .view() method in PyTorch is used to reshape a tensor without changing its underlying data.

        # ( Batch, Seq_Len, d_model) --> ( Batch, Seq_Lem. h_heads, d_k) --> (Batch, h ,Seq_len, d_k)
        # Now, each of the h_heads has its own sequence, represented by only d_k dimensions to process 
        query = query.view(query.shape[0], query.shape[1], self.h_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h_heads, self.d_k).transpose(1,2)

        # Calculate Attention=softmax( QxK(T)/sqrt(dk) ) x V 
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask, self.dropout)

        # NOTE : concatenation part
        # ( Batch, h, Seq_len, d_k ) --> ( Batch, Seq_Len, h, d_k) --> ( Batch, Seq_Len, d_model )
        x = x.transpose(1,2).contigous().view(x.shape[0], -1, self.h * self.d_k)

        # NOTE : last multiplication of M-HA => concat(Matrix) x Wo matrix
        # ( Batch, Seq_Len , d_model ) --> (Batch, Seq_Len, d_model )
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    # sublayer will be a passable layer as an argument ( most likely a FFN layer / something simillar in M-HA )
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # changed forward pass
        # NOTE : original implementation
        # return x + self.dropout(self.norm(sublayer(x)))



# d_model=512
# h_heads=17
# assert d_model % h_heads == 0 , "dasdadas"
# print(f"The result of the operation is {d_model/h_heads}")
a = 7//2
b=7/2
print(a,type(a))
print(b,type(b))
