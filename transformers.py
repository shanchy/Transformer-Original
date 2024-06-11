

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        
        """
        Args:
            d_model : dimension of the embedding vectors
            n_heads : number of self attention heads
            
        Return: N/A
                    
            
        Explanation:
            Why nn.Linear(d_model,d_model)? If d_model=512, that means a giant 512x512 weight matrix.
            If d_model=512, num_heads = 8, then d_head = 64. Shouldn't it be nn.Linear(d_head,d_model)?
            Then for an input I = 1x512, and weight matrix W = 64x512, doing I x W.transpose will give 1x64
            
            Reason is, I=1x512 with W=512x512 will give 1x512. This will be split into 8 parts,1 per head
            That would mean the vector fed to each head would be 1x64.
            So the computation is done one shot for efficiency, instead of multiply with 8 different matrices
            of size 64x512. 
            
        """
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads # Dimension of the Key, Query & Value vector passed to each head
        
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
    
    def scaled_dot_product_attention (self,Q,K,V,mask=None):
        """
        Args:
            Q,K,V : Query, Key & Value matrices. Dimension is (batch_size x num_heads x seq_length x d_head)
            mask : masking locations for the decoder
            
        Return:
            Z : Context vectors, from multiple self-attention blocks
                Dimenesion is (batch_size x num_heads x seq_length x d_head)
        """
        
        #Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_head)
        
        #Apply mask if available
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
            
        #Compute softmax, row-wise, i.e. for a given row all column values are fed to the softmax
        attn_probs = torch.softmax(attn_scores,dim=-1)
        
        #Compute output as summation of weighted V vectors
        Z = torch.matmul(attn_probs,V)
        return Z
    
    def split_heads(self,x):
        """
        Args:
            x : A tensor representing either Q,K or V, obtained after applying the corresponding W_q, W_k or
                W_v to the input batch. Size is (batch_size x seq_length x d_model)
        Return: A reordered tensor of size (batch_size x num_heads x seq_length x d_head), which was originally
                (batc_size x seq_length x num_heads x d_head). This was resized from the input.
                num_heads x d_head = d_model
        """
        batch_size, seq_length, d_model = x.size()
        return x.view (batch_size, seq_length, self.num_heads, self.d_head).transpose(1,2)
    
    def combine_heads(self,x):
        """
        Args:
            x : A tensor representing the computed context vectors 
                Dimensions are (batch_size x num_heads x seq_length x d_head)
        Return: This function does the reverse operation of split_heads. Therefore, it takes the input and 
                creates (batch_size x seq_length x d_model)
        """
        batch_size, _, seq_length,d_head = x.size()
        return x.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model)
    
    
    def forward(self,Q,K,V, mask=None):
        """
        Args:
            Q,K,V : All there parameters get a copy of the input batch as an argument.
                    Dimension of input batch is (batch_size x seq_length x d_model)
            mask : masking locations - padding if encoder, and padding+look-ahead if decoder
        Return:
            output : A tensor representing the output of the multi-head self-attention block
                    Dimension is (batch_size x seq_length x d_model)
        
        
        Explanation:
            Input is a batch of samples. 
            Assume batch_size = 32. Each sample is a sequence of tokens. Assume seq_length = 16. Each token
            will be converted to a embedding. Assume embedding size = 512. 
            Then, after the embedding is generated for a batch of 32 samples and the positional embeddings are
            added, each batch has dimensions (32x16x512) 
            
            ========== Compute the Q, K & V vectors ============
            
            The weight matrices W_q, W_k & W_v are all 512x512 (see reason in __init__ method). 
            Q, K & V are computed by apply the corresponding weight matrices to a copy of the input batch. 
            The resultant matrix's dimensions remains same as the input batch at 32x16x512.
            
            Each matrix is then resized to (32x16x8x64). This means that for each of the 16 tokens, there are 
            8 vectors, each of them 64-D, which will be fed in parallel to the 8 attention heads.
            For efficient computation, each matrix is rearranged to (32x8x16x64). Now, for each of the 8 heads,
            there are 16 vectors (corresponding to 16 tokens in each sequence) of 64-D each.
            In other words, each of the 8 attention heads will receive a batch of 32 tensors, where each 
            tensor will consist of 16 vectors, each of dimension 64-D.
        
            ========== Compute multi-head self-attention context vectors =========
            Attention scores are computed by multiplying K = (32x8x16x64) by transpose of Q = (32x8x64x16) 
            Note that prior to transpose it was (32x8x16x64). Resultant matrix is 32x8x16x16
            This score is rescaled by dividing with sqrt(d_head), where d_head = 64. Next softmax is applied.
            Dimensions don't change during the rescaling and softmax
            Next the matrix is multiplied by the value matrix. So (32x8x16x16) x (32x8x16x64) -> (32x8x16x64).
            This multiplication perform 2 actions to compute the final context vector for each token,: 
            a) it multiplies the V vectors with the computed attention scores, and
            b) sums the vectors.
            This is done across all 8 heads in parallel, generating 8 context vectors (64-D) per token 
            
            ========== Compute combined output ===========
            The (32x8x16x64) is reversed back to (32x16x8x64) which is (batch_size x seq_length x num_heads
            x d_head). It is then rearranged so that for each token, the 8 vectors of 64-D dimension are 
            concatenated to form a 512 vector, resulting in (32x16x512). This is then passed through a liner
            layer with weight matrix 512x512. The final tensor is (32x16x512)
            
        """
    
        # Compute the Q, K & V vectors 
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Compute multi-head self-attention context vectors
        attn_ctxt_vecs = self.scaled_dot_product_attention(Q,K,V, mask)
        
        # Compute combined output
        output = self.W_o(self.combine_heads(attn_ctxt_vecs))
        
        return(output)
           

class PositionWiseFF(nn.Module):
    def __init__(self,d_model,d_ff):
        """
        Args:
            d_model : dimension of the multi-head self-attention output vector
            d_ff : dimension of the inner Linear layer, usually much smaller than d_model
            
        Return: N/A
        """
        super(PositionWiseFF,self).__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        """
        Args:
            x : output of the multi-head self-attention block. Size is (batch_size x seq_length x d_model) 
            
        Return: Tensor, the same size as the input x
        
        Explanation:
            Position-wise means that every single token representation will be fed to the neural net.
            Feed-forward networks usually comprise of two Linear layers. The first one expands the dimension and
            the second one reduces it back
            
            It is important to note that even though the input d_model is exactly the same as the input 
            embedding dimension, this is not compulsory. Depending on the weight matrices in the attention 
            block, this d_model dimension could be smaller/larger compared to the input embedding dimension.
            Similarly, the second layer reduces back to the same size as the input. This is also not compulsory
            For the sake of simplicity, in this architecture, dimension of d_model is used everywhere.
        """
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_length):
        """
        Args:
            d_model : input embedding dimension
            max_seq_length : maximum number of tokens in an input sample
        
        Return : N/A
        """
        super(PositionalEncoding,self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        
        # A vector representing the index positions of each token in the sequence, reshaped to a column vector
        position = torch.arange(0,max_seq_length,dtype=torch.float).unsqueeze(1)
   
        # A curve that starts at 1 and decays exponentially towards 0, index increases from 0 -> d_model, at 
        # interval of 2
        div_term = torch.exp(torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model))
        
        # Input embedding has dimension d_model.
        # To compute the corresponding positional embedding of similar dimension,
        # For any given even position i in the embedding, two values are computed
        # 1) Value for that position i = Sin of the div_term, based on that particular position i
        # 2) Value of the next position i+1 = Cos of the div_term, based on that same position i
        # The position refers to the position of each token in the sequence.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Unsqueezing helps to make the tensor the same size as the input
        self.register_buffer('pe',pe.unsqueeze(0))
        
    def forward(self,x):
        """
        Args:
            x : input embeddings.Size is (batch_size x seq_length x d_model)
        
        Return: Tensor, the same size as x
        
        Explanation:
        
            The original positional encoding formula is as follows:
            PE(pos,2i)     = sin(pos/10000^((2*i)/d_model))
            PE(pos,2i + 1) = cos(pos/10000^((2*i)/d_model))
        
            In this code, the following formula is used
            PE(pos,2i)     = sin(pos/e^((i*log(10000))/d_model))
            PE(pos,2i + 1) = cos(pos/e^((i*log(10000))/d_model))
            
            So the primary difference is that for the denominator,
            e^((i*log(10000))/d_model) is used instead of 10000^((2*i)/d_model).
            However, both of them have similar exponential decay behaviour from 1 towards 0 with as
            index increase from 0 -> d_model.
            
            So the overall behaviour is tha the sine will approach 0 and the cos will approach 1.
            The pos value increases the frequency and therefore controls how many oscillations occur before
            the curves approach 0/1. 
            
            It is this difference in frequency that makes each curve unique and hence gives each token a unique
            embedding based on the token position!
            
            Note:
            The div_term can also be computed as follows.
            div_term = 1/torch.exp((torch.arange(0, 50, 2).float() * math.log(10000.0))/50)
            
        """
        return x + self.pe[:,:x.size(1)] # Sequence length is the second dimension (dim=1) of x
  
    

class EncoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        """
        Args:
            d_model : dimension of the input samples
            num_heads : number of attention heads
            d_ff : number of nodes in the hidden layer of the feed forward network
            droput : dropout ratio
            
        Return: N/A
        """
        super(EncoderBlock,self).__init__()
        self.mult_attn = MultiHeadAttention(d_model,num_heads)
        self.feed_forward = PositionWiseFF(d_model,d_ff)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self,x,mask):
        """
        Args: 
            x : Input to the encoder. Size is usually (batch_size x seq_length x d_model)
            mask : specifies which token positions in the sequence are padded

        Return:
            x : an output tensor, similar in size to the input x
            
        Explanation:
            Each encoder block has 2 main components
            a) The multi-head self attention
            b) The feed forward network
            
            First, the multi-head attention is computed for the encoder input. Any necessary padding masks are
            specified. Add and norm is performed on the output
            This is then fed to the position-wise feed forward network. Add and norm is peformed on the output
        """
        attention_output = self.mult_attn(x,x,x,mask)
        x = self.norm_1(x + self.dropout(attention_output))
        feedforward_output = self.feed_forward(x)
        x  = self.norm_2(x + self.dropout(feedforward_output))
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        """
        Args:
            d_model : dimension of the input samples
            num_heads : number of attention heads
            d_ff : number of nodes in the hidden layer of the feed forward network
            droput : dropout ratio
            
        Return: N/A
        """
        super(DecoderBlock,self).__init__()
        self.mult_attn = MultiHeadAttention(d_model,num_heads)
        self.cross_attn = MultiHeadAttention(d_model,num_heads)
        self.feed_forward = PositionWiseFF(d_model,d_ff)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,encoder_output, source_mask, target_mask):
        """
        Args: 
            x : Input to the decoder. Size is usually (batch_size x seq_length x d_model)
            encoder_output : Output from the final encoder block. Size is (batch_size x seq_length x d_model)
            source_mask : specifies which token positions in the input to encoder is padded
            target_mask : specifies the look-ahead masks for the input to the decoder 

        Return:
             : an output tensor, similar in size to the input x
             
        Explanation:
            Each decoder block has 3 main components
            a) The masked multi-head self attention
            b) The normal multi-head attention
            c) The position-wise feed forward network
            
            First, the masked multi-head attention is computed for the decoder input. During training, 
            the decoder input is the target sequence, but specified with the look-ahead mask. The look-ahead
            mask ensures that for each current token in the target sequence, it only has access to information 
            from preceding tokens. Succeeding tokens are masked and hence cannot contribute to the computations
            of the current token.             
            Add and norm is performed on the output of the masked multi-head attention.
            Next, the output is fed to a cross multi-head attention and used to compute K. 
            This cross multi-head attention also receives the outputs from the final encoder block, which
            are used to compute Q & V.
            Add and norm is performed on the output of the cross multi-head attention.
            This is then fed to the position-wise feed forward network. Add and norm is peformed on the output
        """
        attention_output = self.mult_attn(x,x,x,target_mask)
        x = self.norm_1(x + self.dropout(attention_output))
        cross_attention_output = self.cross_attn(x,encoder_output,encoder_output,source_mask)
        x = self.norm_2(x + self.dropout(cross_attention_output))
        feedforward_output = self.feed_forward(x)
        x  = self.norm_3(x + self.dropout(feedforward_output))
        return x
        
        

class Transformer(nn.Module):
    def __init__(self,source_vocabsize,target_vocabsize,
                 d_model,num_heads,num_layers,d_ff,max_seq_length,droput):
        """
        Args:
            source_vocabsize : size of the source vocabulary
            target_vocabsize : size of the target vocabulary
            d_model : dimension of input embedding
            num_heads : number of self-attention heads
            num_layers : number of encoder and/or decoder blocks
            d_ff : number of nodes in the inner layer of the feed-forward network
            max_seq_length: max length of an input/output sequence
            droput : dropout ratio

        Return: N/A
        
        """
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(source_vocabsize, d_model)
        self.decoder_embedding = nn.Embedding(target_vocabsize, d_model)
        self.positional_encoding = PositionalEncoding(d_model,max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model,num_heads,d_ff,droput) 
                                            for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model,num_heads,d_ff,droput) 
                                            for _ in range(num_layers)])
        self.fc = nn.Linear(d_model,target_vocabsize)
    
    def generate_mask(self,source,target,pad_idx):
        """
        Args:
            source: Input tensor of size (batch_size x sequence_length). Each row consists of a sequence of 
                    vocabulary indices, which is used to obtain embeddings. If the number of tokens in the 
                    sequence is less than "sequence_length", the remaining positions have a <pad> token, represented
                    by pad_idx, usually 0
                    
                    Eg, 
                    sequence length is 16.
                    let sample be the following sentence "I like cats and dogs, but I prefer dogs". Length = 10.
                    Therefore, the remaining 6 positionos are padded as follows
                    I like cats and dogs , but I prefer dogs <pad> <pad> <pad> <pad> <pad> <pad>
                    Converted to indices: 32 458 9128 27 9456 5 46 32 7321 9456 0 0 0 0 0 0
                    
            target: Input tensor of size (batch_size x sequence_length). Details, similar to source
            pad_idx : index value representing the pad token

        Return:
            source_mask: Tensor of boolean values, where padded positions are masked. The tensor will have 
                         dimensions such that it can be applied to the encoder multi-head attention scores
                         Dimension will be (batch_size x 1 x 1 x sequence_length). See Explanation
            target_mask: Tensor of boolean values, where padded and look-ahead positions are masked. Tensor
                         dimensions must match the masked multi-head attentions scores and the encoder-decoder
                         multi-head attention scores.
                         Dimension will be (batch_size x 1 x sequence_length x sequence_length). See Explanation
                         
        Explanation:
            Let's assume a batch size of 3, with sequence length of 5. 
            So input size is (batch_size x sequence_length) = (3 x 5)
            
            source = [[35, 87, 234, 1233, 0],
                      [56, 12, 231, 2323, 722], 
                      [9121, 444, 0, 0 ,0]]
                      
            target = [[723, 823, 31, 0, 0],
                      [981, 323, 3095, 31, 0],
                      [91, 8756, 8123, 231, 88]]
                      
            =====================================          
            First, the source_mask needs to be obtained. The source values are converted to boolean, where False indicates
            masked positions, as follows
            [[ True,  True,  True,  True, False],    <-- sample 1
             [ True,  True,  True,  True,  True],    <-- sample 2
             [ True,  True, False, False, False]].   <-- sample 3
             
             This tensor is the same size as the input, which is (3 x 5). However it needs to be adjusted so that
             it can be applied to the attention scores.
             Assuming, there are 8 heads, the tensor of attention scores will be (3 x 8 x 5 x 5)
             
             Performing unsqueeze(1), follwed by unsqueeze(2), converts the (3x5) mask to (3 x 1 x 1 x 5).
             Now this can be applied to the attention scores and is returned by the function
            
             BUT HOW WILL IT BE APPLIED TO THE ATTENTION SCORES???
             
             Let's look at sample 1 mask = True, True, True, True, False. This means the 5th position is masked
             Remember that for each sample, there are 8 heads, and each head has attention scores of 5x5,
             and hence the last three dimension of 8x5x5 in (3x8x5x5).
             During the mask application, the boolean value is broadcasted across the second dimension to become
             
             [[True, True, True, True, False],
              [True, True, True, True, False],
              [True, True, True, True, False],
              [True, True, True, True, False],
              [True, True, True, True, False]].
              
              Do notice how the 5th position that was padded always gets masked out.
              
              This is then broadcasted across the 3rd dimension, so that each of the heads for the 1st sample
              gets this same 5x5 tensor as the mask.
              
              This broadcasting occurs in this manner due to the dimensionality of 1 in (3 x 1 x 1 x 5).
              
              In similar fashion, the 2nd sample is as follows after broadcasting
              
              [[ True,  True,  True,  True,  True],
               [ True,  True,  True,  True,  True],
               [ True,  True,  True,  True,  True],
               [ True,  True,  True,  True,  True],
               [ True,  True,  True,  True,  True]]
              
               and the 3rd sample is 
               [[ True,  True, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True, False, False, False]]
             
                Take not how the 2nd sample has no padding, whereas the 3rd sample always has the 3rd - 5th
                positions padded.
                
                ============================
                Next, the target mask needs to obtained.
                The target mask goes through the same masking process for padded tokens as the source mask, 
                but, it also has to go through look-ahead masking.
                
                When applying masking to the target for the padded tokens, the target_mask is as following.
                
                [[ True,  True,  True,  False, False],    <-- sample 1
                 [ True,  True,  True,  True,  False],    <-- sample 2
                 [ True,  True,  True,  True,  True]].    <-- sample 3
                 
                Similar to the source, this is (3 x 1 x 1 x 5) and will be applied to the attention scores via
                broadcasting.
                
                The look head mask, called the nopeak_mask is based on the sequence length is obtained by
                a function that applies masks diagonally. In this case, the sequence length is 5, hence
                nopeak_mask is a follows:
                
                [[ True, False, False, False, False],  
                 [ True,  True, False, False, False],
                 [ True,  True,  True, False, False],
                 [ True,  True,  True,  True, False],
                 [ True,  True,  True,  True,  True]]
                 
                 Notice the diagonal nature. When this mask is applied to the attention score of (5x5),it means
                 that at any given position in that sequence, the tokens only have access to information from
                 itself and previous tokens, not the tokens after it. During training, the target is actually fed
                 to the decoder, and therefore this masking is mandatory because the model should not be able to 
                 see the tokens that it is going to predict. 
                 
                 The final target mask is obtained by doing target_mask & nopeak_mask
                 Note that this nopeak_mask can be (5x5) or (1x5x5). Either way it is compatible with the
                 target_mask which is (3x1x1x5) 
                 
                 Do notice that during this & operation, the values in the 1st dimension of the target_mask
                 are broadcasted across the second dimension to get (3x1x5x5) to match(1x5x5).
                 
                 After the & operation, the resultant target mask becomes a (3x1x5x5):
                 
                 [[[[ True, False, False, False, False],
                    [ True,  True, False, False, False],     
                    [ True,  True,  True, False, False],     5x5 tensor for sample 1, applied to all 8 heads
                    [ True,  True,  True, False, False],     Notice the last 2 positions are always masked,
                    [ True,  True,  True, False, False]]],   in addition to the look-ahead mask


                  [[[ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],     5x5 tensor for sample 2, applied to all 8 heads
                    [ True,  True,  True,  True, False],     Notice the last position is always masked,
                    [ True,  True,  True,  True, False]]],   in addition to the look-ahead mask


                  [[[ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],    5x5 tensor for sample 3 applied to all 8 heads
                    [ True,  True,  True,  True, False],    Notice no position is always masked. 
                    [ True,  True,  True,  True,  True]]]]  Only the look-ahead mask exists
                    
                This (3x1x5x5) tensor is the target_mask is returned by the function
                     
        """
        source_mask = (source != pad_idx ).unsqueeze(1).unsqueeze(2)
        # target_mask = (target != pad_idx ).unsqueeze(1).unsqueeze(3) # might be incorrect. see reason below
        # unsqueeze(1).unsqueeze(3) might be incorrect for padding as it's different from unsqueeze(1).unsqueeze(2)
        # so use the first one instead
        target_mask = (target != pad_idx ).unsqueeze(1).unsqueeze(2)
        seq_length = target.size(1)
        
        #Note that target_mask is (batch_size x 1 x 1 x seq_length) so,
        #torch.ones(seq_length, seq_length) that will generate (seq_length x seq_length) instead of 
        #(1 x seq_length x seq_length) will also work!
        nopeak_mask = (1 - torch.triu(torch.ones(1,seq_length,seq_length),diagonal=1)).bool()
        target_mask = target_mask & nopeak_mask
        return source_mask, target_mask
    
    def encode(self, source, source_mask):
        """
        Args:
            source : Input of size (batch_size x sequence_length), where each row is a sequence of token ids
            source_mask : A tensor where padded positions in the source are masked. This tensor is resized
                          to match the multi-head attention scores. See function generate_mask for example
        
        Return:
            encoder_output : Output from the last encoder block. 
                             Size is (batch_size x sequence_length x d_model)
        """
        
        # Obtain the source embeddings based on the ids
        source_embed = self.positional_encoding(self.encoder_embedding(source))
        
        # Execute the sequence of encoder blocks
        # First encoder block takes the source_embeddings as input and subsequent encoder blocks take the 
        # previous encoder block's output as input
        encoder_output = source_embed
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output,source_mask)
        
        return encoder_output
    
    def decode(self, target,encoder_output,source_mask,target_mask):
        """
        Args:
            target: Input of size (batch_size x sequence_length), where each row is a sequence of token ids
            encoder_output : Output from the encoder function
            source_mask : A tensor where padded positions in the source are masked. This tensor is resized
                          to match the multi-head attention scores. See function generate_mask for example
            target_mask : A tensor where padded positions & look-ahead positions in the source are masked. 
                          This tensor is resized to match the multi-head attention scores. 
                          See function generate_mask for example
        Return: 
            decoder_output : Output from the last decoder block. 
                             Size is (batch_size x sequence_length x d_model)
        
        """
        # Obtain the target embeddings based on the ids
        target_embed = self.positional_encoding(self.decoder_embedding(target))
        
        # Execute the sequence of decoder blocks
        # For the first decoder block, the masked multi-head attention takes the target_embedding as input
        # and the subsquent decoder blocks take the previous decoder block's output as input
        # For each decoder block, the encoder-decoder or cross multi-head attention takes the output from the 
        # final encoder block to calculate  Q & V values, and the output from the previous decoder block 
        # to calculate the K values
        decoder_output = target_embed
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output,encoder_output,source_mask,target_mask)
        
        return decoder_output
    
        
    def do_inference(self, src, src_mask, sos_idx, eos_idx, max_len=50):
        """
        Args:
            src : Input of size (batch_size x sequence_length), where each row is a sequence of token ids
            src_mask : A tensor where padded positions in the source are masked. This tensor is resized
                       to match the multi-head attention scores. See function generate_mask for example
            sos_idx : index of the sos token
            eos_idx : index of the eos token
            max_len : maximum sequence length
            
        Return:
            preds : Output of size (batch_size x sequence_length), where each row is a sequence of token ids
            
        
        """
        
        # Obtain the number of samples
        batch_size = src.shape[0]
        
        # Create an array of booleans that indicates whether a particular sample/sequence has finished decoding
        # Initialize to False, and assign True when done decoding
        done = {i:False for i in range(batch_size)}
        
        # Set to evaluation mode
        self.eval()
        
        # Initialize the predictions for all samples with the index of the SOS token
        preds = torch.LongTensor([[sos_idx]] * batch_size).to(src.device)
        
        # Run the samples through the transformer encoder and obtain the encoder outputs
        encoder_output = self.encode(src, src_mask)
        
        # Run the decoder iteratively and obtain the index of the next token in each iteration, until the 
        # EOS token is obtained.
        for _ in range(max_len-1):
            
            # Run the decoder based on the encoder_outputs & all the predictions till the current iteration
            decoder_output = self.decode(preds,encoder_output,src_mask,None)
            
            # Obtain the logits based on the final fully connected layer
            logits = self.fc(decoder_output)
            
            # The dimension of the logits is (batch_size,sequence_length,vocab_size)
            # For each sample in the batch, take the last row and find the index corresponding to the max value 
            # in that row. That is the index of the next token.
            next_idx = torch.max(logits[:,-1:],dim=-1).indices
            
            # Update the predictions by appending these indices
            preds = torch.concat((preds,next_idx),dim=1)
            
            # Mark a sample as done if index of EOS token is obtained in current iteration.
            # Exit loop if all samples are marked as done
            for i,idx in enumerate(next_idx):
                if idx[0] == eos_idx:
                    done[i] = True
            if False not in done.values():
                break
               
        # Set to training mode
        self.train()
        return preds
            
    def count_parameters(self):
        """
        Args: N/A
        
        Return : N/A
        
        Explanation:
        
            This function calculates the number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    def forward(self,source,target,pad_idx):
        """
        Args:
            source: Input tensor of size (batch_size x sequence length). Consists of index ids used to obtain
                    embedding
            target: Input tensor of size (batch_size x sequence length). Same indices as above.

        Return:
            output: Logits of size (batch_size x sequence length x target_vocabulary). This is then fed to 
                    softmax, so that for any given sequence in the batch, the next word at any position in 
                    that sequence can be obtained
        """
        # Get the source_mask which masks out padded locations and get the target_mask which masks padded
        # locations and look-ahead locations
        source_mask,target_mask = self.generate_mask(source,target,pad_idx)
        
        # Execute the encoder
        encoder_out = self.encode(source,source_mask)
        
        # Execute the decoder
        decoder_out = self.decode(target,encoder_out,source_mask,target_mask)
        
        # Final layer is a fully connected layer with the size of the target_vocabulary
        # It outputs logits which are converted to probabilities by the Softmax portion of the CrossEntropy.
        # Those proabibilities are then used to predict the next word/token at each position.
        output = self.fc(decoder_out)
        
        return output

from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Specify the BOS,SOS & PAD indices
bos_idx, eos_idx, pad_idx = 1, 2, 0
# Use 100 integers and max length of 16 only.
vocab_size, src_len = 100, 16

# Create a data loaoder.
data_loader = data.DataLoader( 
    # Create a dataset of number sequences, where each sequence can be 8 - 16 numbers long
    dataset=[torch.randint(3,vocab_size,(torch.randint(src_len//2,src_len,(1,)),)) for _ in range(50000)],
    # Specify batch size, shuffle the samples and drop the last batch if it is not equal to batch_size
    batch_size=128, shuffle=True, drop_last=True,
    # For each sample in a batch, create the source and target sequences for training
    # For the source, take each sample in the batch and pad with 'pad_idx', if sequence length is less than 16
    # For the target, reverse each sample in the batch, encapsulate with 'bos_idx' & 'eos_idx', and do padding
    collate_fn=lambda batch: (
        # SOURCE SAMPLES FOR THE BATCH
        pad_sequence(batch, batch_first=True, padding_value=pad_idx), 
        
        # TARGET SAMPLES FOR THE BATCH
        pad_sequence([torch.LongTensor([bos_idx] + x.flip(0).tolist() + [eos_idx]) for x in batch],
            batch_first=True, padding_value=pad_idx,)),
    )

# Create a small transformer as this is a simple task. Start by specifying the necessary transformer parameters
source_vocabsize = 100
target_vocabsize = 100
d_model = 64
num_heads = 2
num_layers = 2
d_ff = 128
max_seq_length = 32
dropout = 0.1

# Instantiate the transformer
seq_reverser = Transformer(source_vocabsize,target_vocabsize,d_model,num_heads,num_layers,
                         d_ff,max_seq_length,dropout)
# Instantiate the optimizer
optim = torch.optim.AdamW(seq_reverser.parameters(), lr=1e-3, weight_decay=1e-4)

steps_per_epoch = len(data_loader)
# Train the transformer for 15 epochs.
num_epochs = 15
for epoch_index in range(num_epochs):
    running_train_loss = 0
    i = 0 # number of training iterations per epoch
    for src, tgt in data_loader:
        src, tgt = src.to(device), tgt.to(device)
        # Previous outputs are fed back as input into the decoder, and the transformer keeps predicting the 
        # next output in the sequence until the EOS is reached.
        # tgt_in is fed into the decoder and tgt_out is compared against the actual transformer outputs
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        # Forward propagation
        logits = seq_reverser(src, tgt_in, pad_idx) 
        # Calculate the loss
        loss = nn.functional.cross_entropy(logits.permute(0,2,1), tgt_out, ignore_index=pad_idx)

        optim.zero_grad()
        # Backward Propagation of errors
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seq_reverser.parameters(), 1.)
        # Update model parameters
        optim.step()
        
        i=i+1 
        print (f"Completed training step: {i}/{steps_per_epoch} in epoch: {epoch_index+1}.Training loss: {loss}",end='\r')
        running_train_loss+=loss.item()
    print("Training loss : ",running_train_loss/steps_per_epoch)

x = torch.LongTensor([56, 71, 23,44, 91, 92, 93, 55, 56,57,83,71,33 ]).unsqueeze(dim=0).to(device)
y = seq_reverser.do_inference(x, None, bos_idx, eos_idx, max_len=32)
print("Input   = ", x) # Print the input sequence
print("Output  = ",y[:,1:-1]) # Prin the reversed sequence, but exclude BOS & EOS indices
Input   =  tensor([[56, 71, 23, 44, 91, 92, 93, 55, 56, 57, 83, 71, 33]])
Output  =  tensor([[33, 71, 83, 57, 56, 55, 93, 92, 91, 44, 23, 71, 56]])

# Install the following packages, if any of them fail during import
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter

# Load the german & english datasets from Multi30k as source and target sets, respectively
train_data = Multi30k(root="datasets", split="train", language_pair=("de","en"))
# Convert the training set to a list of tuples. Each tuple has a german sentence as source & english sentence
# as target
train_data = [(src, tgt) for src, tgt in train_data if len(src) > 0]

UNK, PAD, BOS, EOS = ("<UNK>", "<PAD>", "<START>", "<END>") # Specify the special tokens

# this tokenizer just splits the sentence into words & converts upper caps to lower caps
tokenizer = get_tokenizer("basic_english") 

# For each language, create a counter to count the number of occurences of each word
en_counter, de_counter = Counter(), Counter() 

# Go through the german and english sentences & update their respective counters
for src, tgt in train_data:
    de_counter.update(tokenizer(src))
    en_counter.update(tokenizer(tgt))

# Create the vocabularies of each language based on the counters & the special tokens
# The vocabulary is a {word,index} dictionary
de_vocab = vocab(de_counter, specials=[UNK, PAD, BOS, EOS])
de_vocab.set_default_index(de_vocab[UNK])
en_vocab = vocab(en_counter, specials=[UNK, PAD, BOS, EOS])
en_vocab.set_default_index(en_vocab[UNK])
pad_idx = de_vocab[PAD] # pad_idx is 1
assert en_vocab[PAD] == de_vocab[PAD]


# Create the dataloader
train_loader = data.DataLoader(
    dataset=train_data,
     # Specify batch size, shuffle the samples and drop the last batch if it is not equal to batch_size
    batch_size=128, shuffle=True, drop_last=True,
    # For each sample in a batch, create the source and target sentences for training
    # For the source, take each sample in the batch, tokenize it and then convert it to a sequence of indices,
    # and pad with 'pad_idx', if sequence length is less than the longest sentence in that batch
    # For the target, take each sample in the batch, and apply all operations as the source, but encapsulate
    # the sequence with the BOS & EOS indices before padding
    collate_fn=lambda batch: (
        pad_sequence(
            [torch.LongTensor(de_vocab(tokenizer(x))) for x, _ in batch],
            batch_first=True, padding_value=pad_idx),
        pad_sequence(
            [torch.LongTensor(en_vocab([BOS] + tokenizer(y) + [EOS])) for _, y in batch],
            batch_first=True, padding_value=pad_idx),
    ),
    num_workers=4,
)

# Create a transformer for the machine translation. It will be slightly larger than the simple sequence
# reverser, but still extremely small compared to LLMs
d_model = 256
num_heads = 8
num_layers = 4
d_ff = 512
max_seq_length = 256
dropout = 0.1
source_vocabsize = len(de_vocab) # approximately 30k
target_vocabsize = len(en_vocab) # approximately 30k

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the transforer
ger_eng_translator = Transformer(source_vocabsize,target_vocabsize,
                          d_model,num_heads,num_layers,d_ff,max_seq_length,dropout)

ger_eng_translator.to(device)
# Instantiate the optimizer
optim = torch.optim.AdamW(ger_eng_translator.parameters(), lr=1e-4, weight_decay=1e-4)


steps_per_epoch = len(train_loader)
# Train the transformer for 60 epochs
num_epochs = 60
for e in range(num_epochs):
    running_train_loss = 0
    i = 0 # number of training iterations per epoch
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        # Previous outputs are fed back as input into the decoder, and the transformer keeps predicting the 
        # next output in the sequence until the EOS is reached.
        # tgt_in is fed into the decoder and tgt_out is compared against the actual transformer outputs
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        # Forward propagation
        logits = ger_eng_translator(src, tgt_in, pad_idx)`
        # Loss calculation
        loss = nn.functional.cross_entropy(logits.permute(0,2,1), tgt_out, ignore_index=pad_idx)

        # Backward propagation
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ger_eng_translator.parameters(), 1.)
        # Update model parameters
        optim.step()
        
        i=i+1 
        print (f"Completed training step: {i}/{steps_per_epoch} in epoch: {e+1}.Training loss: {loss}",end='\r')
        running_train_loss+=loss.item()
    print("Training loss : ",running_train_loss/steps_per_epoch)

sentence_1 = 'zwei frauen spazieren und lachen im park.'
sentence_2 = 'ein Mann, der mit seinem Hund spazieren geht.'
sentence_3 = 'ein Frau in schwarzer Jacke, der auf einem weißen Pferd reitet.'
sentence_4 = 'Fünf Wanderer, von denen einer in Richtung Kamera blickt und die anderen von der Kamera weg, gehen durch ein steiniges Flussbett.'
sentence_5 = 'Ein Mann isst eine Orange, während er mit seinem Sohn spricht.'

testSentence = sentence_3

x = torch.LongTensor(de_vocab(tokenizer(testSentence))).unsqueeze(dim=0).to(device)
y = ger_eng_translator.do_inference(x, None, en_vocab[BOS], en_vocab[EOS])

# Print the German sentence
print("German : ")
print(testSentence)

print()

