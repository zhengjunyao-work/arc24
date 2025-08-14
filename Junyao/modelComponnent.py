
class myModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(myModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.multi_head_attention = Attention(embed_dim=input_dim)
        self.single_head_attention = Attention(embed_dim=input_dim)
        self.outLinear = nn.Linear(output_dim, output_dim)
    def forward(self, x):

        x = self.linear(x)
        x = self.multi_head_attention(x)
        x = self.single_head_attention(x)
        x = self.outLinear(x)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim):
        self.ebed_dim = embed_dim
        self.head_dim = embed_dim // 8  # Assuming 8 heads, adjust as necessary
        self.scale = self.head_dim ** 0.5  # Scale factor for attention

    def single_head_attention(self, x):
        self.scale =  (self.head_dim ** 0.5) # Scale factor for attention scores this is to remove variance  to keep variance constant at 1 
    #attention mechanism        self.attention = nn.Sequential(
        embed_dim = 128
        #you have three matrix to tune , key ,query ,value,
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
                #then create query key, value vector and check  how close they are to each other.
        Q,K,V = self.query(x), self.key(x), self.value(x)
        attn_scores = torch.bmm(Q,K.transpose(1,2))/(self.scale) # this is batch matrix multiplication
        attn_weights = F.softmax(attn_scores, dim=-1)  # Normalize scores to probabilities
        out = torch.bmm(attn_weights, V)  # Weighted sum of values
        return out
    def multi_head_attention(self, x):
        # multiple head means splitting the embed dimension into multiple parts and applying the attention mechanism independently on each part
        num_heads = 8
        self.head_dim = embed_dim // num_heads
        Q = self.query(x).view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2)

        # then compute the scores and weights for each head
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) /self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)  # Weighted sum of values for each head
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim) # this gives final output of an attention mechanism 

        return output 
    




def main():
    # get an optimizier. 
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # get a customize loss function
    def loss_function(recon_x, x):
        return sum(abs(recon_x,x))
    model = myModel(input_dim=128, output_dim=10)
    #now we can train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_x = model(data)
        loss = loss_function(recon_x, data) 
    