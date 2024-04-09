import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

char_list = ['-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<sos>', '<eos>', '<pad>']
char_to_num = {c:i for i, c in enumerate(char_list, start=0)}
num_to_char = {i:c for i, c in enumerate(char_list, start=0)}

class ConvEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1d_1 = nn.Conv1d(1024, 256, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(256, 128, kernel_size=1)
        self.lstm = nn.LSTM(128, output_dim//2, batch_first=True, bidirectional=True, num_layers=2)
        self.batch_norm2d = nn.BatchNorm2d(64)
        self.batch_norm1d = nn.BatchNorm1d(128)
        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.activation = nn.LeakyReLU()
        self.dropout2d = nn.Dropout2d(p=0.1) # 채널에 대하여 dropout
        self.dropout1d = nn.Dropout1d(p=0.1) # 채널에 대하여 dropout
        
        nn.init.kaiming_normal_(self.conv2d_1.weight)
        nn.init.kaiming_normal_(self.conv2d_2.weight)
        nn.init.kaiming_normal_(self.conv2d_3.weight)
        nn.init.kaiming_normal_(self.conv1d_1.weight)
        nn.init.kaiming_normal_(self.conv1d_2.weight)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)
        
    def forward(self, images):
        # (batch, 3, 64, 512)
        x = self.conv2d_1(images)
        x = self.activation(x)
        x = self.max_pool(x)
        # (batch, 32, 32, 256)
        x = self.conv2d_2(x)
        # x = self.batch_norm2d(x)
        x = self.activation(x)
        x = self.max_pool(x)
        # (batch, 64, 16, 128)
        x = self.conv2d_3(x)
        x = self.batch_norm2d(x)
        x = self.activation(x)
        x = self.dropout2d(x)
        # (batch, 64, 16, 128)
        x = x.view(x.size(0), -1, x.size(-1))
        # (batch, 1024, 128)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.batch_norm1d(x)
        x = self.activation(x)
        x = self.dropout1d(x)
        # (batch, 128, 128)
        x = x.transpose(2,1)
        # (batch, 128, 128)
        x, _ = self.lstm(x)
        # (batch, 128, 64*2)
        return x

    

class Speller(nn.Module):
    def __init__(self, encoder_dim, target_dim, hidden_dim=128, 
                 sos_id=13, eos_id=14, pad_id=15, max_len=20):
        super().__init__()
        self.rnn_layer = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, num_layers=3, dropout=0.5)
        self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.convertor_linear = nn.Linear(encoder_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim*2, target_dim)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(0.5)
        self.emb = nn.Embedding(target_dim, hidden_dim)
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_len = max_len
        
        nn.init.xavier_uniform_(self.rnn_layer.weight_ih_l0)
        nn.init.xavier_uniform_(self.rnn_layer.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn_layer.weight_ih_l1)
        nn.init.xavier_uniform_(self.rnn_layer.weight_hh_l1)
        nn.init.xavier_uniform_(self.convertor_linear.weight)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
    
    def forward_step(self, rnn_in, converted, hidden_state):
        # rnn_input : (batch, 1, hidden_dim*2)
        # converted : (batch, encoder_seq, hidden_dim)
        # rnn_out : (batch, 1, hidden_dim)
        # context : (batch, 1, hidden_dim)
        # att_score : (batch, 1, encoder_seq)
        # concat_out : (batch, 1, hidden_dim*2)
        # step_out : (batch, 1, target_dim)
        rnn_out, hidden_state = self.rnn_layer(rnn_in, hidden_state)
        context, att_score = self.attention(rnn_out, converted, converted)
        concat_out = torch.cat([rnn_out, context], dim=-1)
        
        x = self.dropout(concat_out)
        x = self.output_linear(x)
        
        step_out = self.logsoftmax(x)
        return step_out, hidden_state, context, att_score

    def beam_search(self, encoder_outputs, beam_size=3):
        device = encoder_outputs.device
        batch_size = encoder_outputs.size(0)
        converted = self.convertor_linear(encoder_outputs).tanh() # (batch, encoder_seq, hidden_dim)
        
        target_idxs = []
        for j in range(batch_size):
            # init
            now_top_score = torch.zeros([1]).to(device) # (n,)
            now_top_idx_all = torch.tensor([self.sos_id], dtype=torch.int64).view(1, 1).to(device) # (n, len)
            idx_last = now_top_idx_all # (n, 1)
            hidden_state = None # (1, n, dim)
            context = converted[j:j+1,0:1,:] # (n, 1, dim)
            
            end_top_idx_all = torch.tensor([], dtype=torch.int64).view(0,1).to(device) # (m, len)
            end_top_score = torch.tensor([]).to(device) # (m,)

            for i in range(self.max_len):
                # forward
                
                # hidden_state : (1, n, dim)
                rnn_in = torch.cat([self.emb(idx_last).tanh(), context], dim=-1) # (n, 1, dim*2)
                converted_j = converted[j:j+1,:,:].tile(idx_last.size(0), 1, 1) # (n, encoder_seq, dim)
                step_out, hidden_state, context, _ = self.forward_step(rnn_in, converted_j, hidden_state)
                # step_out : (n, 1, target_dim)
                # hidden_state : (1, n, dim)
                # context : (n, 1, dim)

                score_last, idx_last = step_out.topk(beam_size) # (n, 1, beam)
                score_last = score_last.transpose(2,0).reshape(-1) # (beam*n,)
                idx_last = idx_last.transpose(2,0).reshape(-1,1) # (beam*n, 1)

                # tile
                candidate_score = now_top_score.tile(beam_size) # (beam*n)
                candidate_idx_all = now_top_idx_all.tile(beam_size, 1) # (beam*n, len)
                candidate_context = context.tile(beam_size, 1, 1) # (beam*n, 1, dim)
                candidate_hidden_state = (hidden_state[0].tile(1, beam_size, 1), 
                                          hidden_state[1].tile(1, beam_size, 1)) # (1, beam*n, dim)
                
                # concat
                candidate_idx_all = torch.cat([candidate_idx_all, idx_last], dim=-1) # (beam*n, len+1)
                pad = torch.tensor(self.pad_id).tile(end_top_idx_all.size(0), 1).to(device)
                end_top_idx_all = torch.cat([end_top_idx_all, pad], dim=-1) # (m, len+1)
                
                # beam*n+m
                candidate_score = (candidate_score*i + score_last)/(i+1)
                candidate_score = torch.cat([candidate_score, end_top_score]) # (beam*n+m)
                candidate_idx_all = torch.cat([candidate_idx_all, end_top_idx_all], dim=0) # (beam*n+m, len+1)

                # get mask
                now_size = idx_last.size(0)
                top_mask = torch.BoolTensor(candidate_score.size(0)).to(device) & False
                end_mask = torch.BoolTensor(candidate_score.size(0)).to(device) & False
                end_mask[now_size:] = True
                end_mask[:now_size] = idx_last.reshape(-1) == self.eos_id
                idx_topk = candidate_score.topk(beam_size)[1]
                top_mask[idx_topk] = True
                
                # select
                end_top_idx_all = candidate_idx_all[end_mask&top_mask, :]
                end_top_score = candidate_score[end_mask&top_mask]
                
                no_end_top_mask = (~end_mask[:now_size]) & top_mask[:now_size]
                now_top_score = candidate_score[:now_size][no_end_top_mask]
                now_top_idx_all = candidate_idx_all[:now_size][no_end_top_mask, :]
                idx_last = idx_last[no_end_top_mask, :]
                hidden_state = (candidate_hidden_state[0][:,no_end_top_mask,:],
                                candidate_hidden_state[1][:,no_end_top_mask,:])
                context = candidate_context[no_end_top_mask,:,:]
                
            score = torch.cat([now_top_score, end_top_score]) # (n+m)
            idx_all = torch.cat([now_top_idx_all, end_top_idx_all], dim=0) # (n+m, len+1)
            best_idx_all = idx_all[score.max(dim=0)[1], 1:] # (1, max_len)
            target_idxs.append(best_idx_all)
            
        target_idxs = torch.stack(target_idxs, dim=0) # (batch, max_len)

        return None, target_idxs, None
    
    def greedy_search(self, encoder_outputs):
        device = encoder_outputs.device
        batch_size = encoder_outputs.size(0)
        converted = self.convertor_linear(encoder_outputs).tanh() # (batch, encoder_seq, hidden_dim)
        
        sos = torch.tensor(self.sos_id).tile(batch_size,1).to(device)
        rnn_in = torch.cat([self.emb(sos).tanh(), converted[:,0:1,:]], dim=-1) # (batch, 1, dim*2)
        hidden_state = None
        
        prob_log_seq = []
        target_idxs = []
        att_recode = []
        
        for i in range(self.max_len):
            step_out, hidden_state, context, att_score = self.forward_step(rnn_in, converted, hidden_state)
            argmax = step_out.max(dim=-1)[1] # (batch, 1)
            rnn_in = torch.cat([self.emb(argmax).tanh(), context], dim=-1) # (batch, 1, hidden_dim*2)
            prob_log_seq.append(step_out) # (batch, 1, target_dim)
            target_idxs.append(argmax)
            att_recode.append(att_score) # (batch, 1, encoder_seq)
            
        prob_log_seq = torch.cat(prob_log_seq, dim=1) # (batch, max_len, dim)
        target_idxs = torch.cat(target_idxs, dim=1) # (batch, max_len)
        att_recode = torch.cat(att_recode, dim=1) # (batch, max_len, encoder_seq)
        
        return prob_log_seq, target_idxs, att_recode
    
    def forward(self, encoder_outputs, target_idxs=None):
        if target_idxs is None:
            return self.greedy_search(encoder_outputs)
        # encoder_outputs : (batch, encoder_seq, encoder_dim)
        # target_idxs : (batch, max_len)
        device = encoder_outputs.device
        batch_size = encoder_outputs.size(0)
        converted = self.convertor_linear(encoder_outputs).tanh() # (batch, encoder_seq, hidden_dim)
        
        sos = torch.tensor(self.sos_id).tile(batch_size,1).to(device)
        rnn_in = torch.cat([self.emb(sos).tanh(), converted[:,0:1,:]], dim=-1) # (batch, 1, dim*2)
        hidden_state = None
        
        prob_log_seq = []
        att_recode = []
        
        for i in range(self.max_len):
            step_out, hidden_state, context, att_score = self.forward_step(rnn_in, converted, hidden_state)
            rnn_in = torch.cat([self.emb(target_idxs[:, i:i+1]).tanh(), context], dim=-1) # (batch, 1, hidden_dim*2)
            prob_log_seq.append(step_out) # (batch, 1, target_dim)
            att_recode.append(att_score) # (batch, 1, encoder_seq)
            
        prob_log_seq = torch.cat(prob_log_seq, dim=1) # (batch, max_len, dim)
        att_recode = torch.cat(att_recode, dim=1) # (batch, max_len, encoder_seq)
            
        return prob_log_seq, target_idxs, att_recode


class ImageLAS(nn.Module):
    def __init__(self, encoder_dim=128, decoder_dim=128, target_dim=16, 
                 sos_id=13, eos_id=14, pad_id=15, max_len=20):
        super().__init__()
        self.conv_encoder = ConvEncoder(3, encoder_dim)
        self.speller = Speller(encoder_dim, target_dim, hidden_dim=decoder_dim,
                               sos_id=sos_id, eos_id=eos_id, pad_id=pad_id, max_len=max_len)
        
    def forward(self, inputs, target_idxs=None, use_beam=False):
        # inputs : (batch, 3, 64, 512)
        # target : (batch, 20, 16)
        x = self.conv_encoder(inputs)
        # (batch, 128, 128)
        if use_beam: return self.speller.beam_search(x)
        # (batch, 20, 16), (batch, 20), (batch, 20, 128)
        return self.speller(x, target_idxs)
    
    
class InferenceModel:
    def __init__(self, model_path, sos_id=13, eos_id=14, pad_id=15, 
                 max_len=20, img_h=64, max_img_w=512, device='cpu'):
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_len = max_len
        self.img_h = img_h
        self.max_img_w = max_img_w
        self.device = device
        self.model = self.load_model(model_path)
        self.image_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32), # 0~1
            transforms.Pad([0,0,self.max_img_w,0]),
            transforms.Lambda(lambda x:x[..., :self.max_img_w]),
        ])
    
    def load_model(self, model_path):
        model = ImageLAS(sos_id=self.sos_id, eos_id=self.eos_id, pad_id=self.pad_id, max_len=self.max_len)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model.to(self.device)
    
    def read_image(self, path):
        img = cv2.imread(path)[:,:,::-1] # (h,w,3)
        img = cv2.resize(img, [int(img.shape[1]/img.shape[0]*self.img_h), self.img_h]) # (h,w,3)
        return torch.tensor(img).permute(2,0,1) # (3,h,w)
    
    def __call__(self, img_ndarr_uint8, channel_first=False, is_bgr=False, use_beam=False):
        assert type(img_ndarr_uint8) is np.ndarray
        assert img_ndarr_uint8.dtype == np.uint8
        
        global num_to_char
        
        img = img_ndarr_uint8
        if channel_first: img = np.transpose(img, axes=(1, 2, 0))
        if is_bgr: img = img[..., ::-1]
        
        img = cv2.resize(img, [int(img.shape[1]/img.shape[0]*self.img_h), self.img_h]) # (h,w,3)
        img = torch.tensor(img).permute(2,0,1).to(self.device)
        
        img = self.image_transform(img).unsqueeze(0)
        with torch.no_grad():
            _, y, att_score = self.model(img, use_beam=use_beam)
        idx_seq = y.tolist()[0]
        ch_seq = list(map(lambda x:num_to_char[x], idx_seq))
        string = ''.join(ch_seq).split('<eos>')[0]
        return string, att_score
    
    
    
    
    
    
    
    
    
    