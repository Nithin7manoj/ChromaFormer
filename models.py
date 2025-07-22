import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop

def batch_product(iput, mat2):
		result = None
		for i in range(iput.size()[0]):
			op = torch.mm(iput[i], mat2)
			op = op.unsqueeze(0)
			if(result is None):
				result = op
			else:
				result = torch.cat((result,op),0)
		return result.squeeze(2)


class rec_attention(nn.Module):
	# attention with bin context vector per HM and HM context vector
	def __init__(self,hm,args):
		super(rec_attention,self).__init__()
		self.num_directions=2 if args.bidirectional else 1
		if (hm==False):
			self.bin_rep_size=args.bin_rnn_size*self.num_directions
		else:
			self.bin_rep_size=args.bin_rnn_size
	
		self.bin_context_vector=nn.Parameter(torch.Tensor(self.bin_rep_size,1),requires_grad=True)
	

		self.softmax=nn.Softmax(dim=1)

		self.bin_context_vector.data.uniform_(-0.1, 0.1)

	def forward(self,iput):
		alpha=self.softmax(batch_product(iput,self.bin_context_vector))
		[batch_size,source_length,bin_rep_size2]=iput.size()
		repres=torch.bmm(alpha.unsqueeze(2).view(batch_size,-1,source_length),iput)
		return repres,alpha



class recurrent_encoder(nn.Module):
	# modular LSTM encoder
	def __init__(self,n_bins,ip_bin_size,hm,args):
		super(recurrent_encoder,self).__init__()
		self.bin_rnn_size=args.bin_rnn_size
		self.ipsize=ip_bin_size
		self.seq_length=n_bins

		self.num_directions=2 if args.bidirectional else 1
		if (hm==False):
			self.bin_rnn_size=args.bin_rnn_size
		else:
			self.bin_rnn_size=args.bin_rnn_size // 2
		self.bin_rep_size=self.bin_rnn_size*self.num_directions


		self.rnn=nn.LSTM(self.ipsize,self.bin_rnn_size,num_layers=args.num_layers,dropout=args.dropout,bidirectional=args.bidirectional)

		self.bin_attention=rec_attention(hm,args)
	def outputlength(self):
		return self.bin_rep_size
	def forward(self,single_hm,hidden=None):
		bin_output, hidden = self.rnn(single_hm,hidden)
		bin_output = bin_output.permute(1,0,2)
		hm_rep,bin_alpha = self.bin_attention(bin_output)
		return hm_rep,bin_alpha


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class att_chrome(nn.Module):
    def __init__(self, args):
        super(att_chrome, self).__init__()
        self.n_hms = args.n_hms
        self.n_bins = args.n_bins
        self.ip_bin_size = 1
        
        self.rnn_hms = nn.ModuleList()
        for i in range(self.n_hms):
            self.rnn_hms.append(recurrent_encoder(self.n_bins, self.ip_bin_size, False, args))
            
        self.opsize = self.rnn_hms[0].outputlength()
        
        # --- HM-level encoder for all three models ---
        self.hm_level_rnn_1 = recurrent_encoder(self.n_hms, self.opsize, True, args)
        self.opsize2 = self.hm_level_rnn_1.outputlength()
        
        # --- New: Transformer components ---
        self.model_type = args.model_type
        
        # CORRECTED: The transformer dimension should match the output of the first LSTM level.
        self.transformer_dim = self.opsize # This is 64

        if self.model_type in ['bilstm_transformer', 'bilstm_attention_transformer']:
            self.pos_encoder = PositionalEncoding(d_model=self.transformer_dim, dropout=args.dropout)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.transformer_dim,
                nhead=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                batch_first=False
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
        
        # --- Final linear layers for prediction ---
        self.final_output_dim = self.opsize2
        if self.model_type == 'bilstm_transformer':
            # Transformer output is averaged over the sequence, so the final dim is the transformer's dim
            self.final_output_dim = self.transformer_dim
        elif self.model_type == 'bilstm_attention_transformer':
            # The final representation will be the concatenation of the attention and transformer outputs
            self.final_output_dim = self.opsize2 + self.transformer_dim

        self.fdiff1_1 = nn.Linear(self.final_output_dim, 1)

    def forward(self, iput):
        bin_a = None
        level1_rep = None
        [batch_size,_,_]=iput.size()

        for hm, hm_encdr in enumerate(self.rnn_hms):
            hmod=iput[:,:,hm].contiguous()
            hmod=torch.t(hmod).unsqueeze(2)
            op,a= hm_encdr(hmod)
            if level1_rep is None:
                level1_rep=op
                bin_a=a
            else:
                level1_rep=torch.cat((level1_rep,op),1)
                bin_a=torch.cat((bin_a,a),1)
        
        if self.model_type == 'bilstm_attention':
            level1_rep_for_attention = level1_rep.permute(1,0,2)
            final_rep_1,hm_level_attention_1 = self.hm_level_rnn_1(level1_rep_for_attention)
            final_rep = final_rep_1.squeeze(1)
            
        elif self.model_type == 'bilstm_transformer':
            transformer_input = self.pos_encoder(level1_rep.permute(1,0,2))
            transformer_output = self.transformer_encoder(transformer_input)
            final_rep = torch.mean(transformer_output, dim=0)

        elif self.model_type == 'bilstm_attention_transformer':
            level1_rep_for_attention = level1_rep.permute(1,0,2)
            final_rep_attention, _ = self.hm_level_rnn_1(level1_rep_for_attention)
            final_rep_attention = final_rep_attention.squeeze(1)
            
            transformer_input = self.pos_encoder(level1_rep.permute(1,0,2))
            transformer_output = self.transformer_encoder(transformer_input)
            final_rep_transformer = torch.mean(transformer_output, dim=0)
            
            final_rep = torch.cat((final_rep_attention, final_rep_transformer), dim=1)
        
        else:
            raise ValueError("Invalid model_type specified.")
        
        prediction_m = ((self.fdiff1_1(final_rep)))
        return prediction_m