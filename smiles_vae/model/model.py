import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self,
                 voc,
                 embedding_dim=128,
                 hidden_dim=256,
                 num_layers=1,
                 latent_dim=128,
                 ):
        super().__init__()

        self.bos = voc.vocab['<BOS>']
        self.eos = voc.vocab['<EOS>']
        self.pad = voc.vocab['<PAD>']

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # embed the tokens and define padding index
        self.x_emb = nn.Embedding(num_embeddings=voc.vocab_size,
                                  embedding_dim=embedding_dim,
                                  padding_idx=self.pad,
                                  )

        # define encoder
        self.encoder_rnn = nn.GRU(input_size=embedding_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bidirectional=False,
                                  )
        
        # define mean and variance linear layers
        self.hidden2mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden2logvar = nn.Linear(hidden_dim, latent_dim)
    
        # define decoder
        self.decoder_rnn = nn.GRU(input_size=embedding_dim + latent_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bidirectional=False,
                                  )
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden2vocab = nn.Linear(hidden_dim, voc.vocab_size)

    @property
    def device(self):
        # ensures that other tensors are moved to the same device as the model's parameters
        return next(self.parameters()).device

    def encoder(self, x):
        """
        Encoder step, emulating z ~ E(x) = q_E(z|x)
        x: (batch_size, max_seq_len)
        """
        x_emb = self.x_emb(x)  # (batch_size, max_seq_len, embedding_dim)

        # h_0 defaults to zeros if not provided
        # output of the GRU at each time step is not currently needed
        # h_n is the final hidden state for the input sequence with shape (D * num_layers, batch_size, hidden_dim)
        # such that D=2 if bidirectional (i.e., hidden state for each direction) otherwise D=1
        _, h_n = self.encoder_rnn(x_emb, None)

        # extract the relevant hidden state
        # if unidirectional, take the last hidden state
        # if bidirectional, take the concatenation of the last states from both directions
        h_n = h_n[-(1 + int(self.encoder_rnn.bidirectional)):]
        
        # split h along the batch dimension to create a tuple of tensors where each corresponds to a single batch element
        # concatenate these tensors along the feature dimensions
        # this operation effectively combines the hidden states from different layers and direcitons into one tensor
        # squeeze(0) removes the singleton dimension at the beginning as it's not required for further processing
        h_n = torch.cat(h_n.split(1), dim=-1).squeeze(0)

        # both mu and logvar have shape (batch_size, latent_dim)
        mu = self.hidden2mean(h_n)
        logvar = self.hidden2logvar(h_n)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick is explained here: hhtps://stats.stackexchange.com/a/16338"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = mu + eps * std
        return z

    def decode(self, x, z):
        """
        Decoder step, emulating x ~ G(z)

        Args:
            x: input sequence with shape (batch_size, max_seq_len)
            z: latent vector with shape (batch_size, latent_dim)
        """
        batch_size, seq_len = x.size()

        # add BOS token and remove the las PAD token to keep the length constant
        start_token = torch.full((batch_size, 1), self.bos, dtype=torch.long)
        x = torch.cat((start_token, x[:, :-1]), 1)

        x_emb = self.x_emb(x)

        # repeat z to match max_seq_len
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)  # (batch_size, max_seq_len, latent_dim)

        # concatenate embedding of true sequence to use for teacher forcing
        x_input = torch.cat([x_emb, z_0], dim=-1)  # (batch_size, max_seq_len, hidden_dim + latent_dim)

        # project z up to hidden dimension to create initial hidden state
        h_0 = self.latent2hidden(z)  # (batch_size, hidden_dim)

        # repeat initial hidden state along the number of GRU layers
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)  # (num_gru_layers, batch_size, hidden_dim)

        # focus on the sequence of outputs (hidden states) rather than the final hidden state
        output, _ = self.decoder_rnn(x_input, h_0)
        logits = self.hidden2vocab(output)  # (batch_size, max_seq_len, vocab_size)

        return logits

    def forward(self, x):
        # encode the input sequence
        mu, logvar = self.encode(x)

        # reparameterize the latent space
        z = self.reparameterize(mu, logvar)  # (batch_size, latent_dim)

        # decode the latent vector to reconstruct the input
        recon_x = self.decode(x, z)  # (batch_size, max_seq_len, vocab_size)

        return recon_x, mu, logvar

    def sample_z_prior(self, n_batch):
        """
        Sample z ~ p(z) = N(0, I)
        
        Args:
            n_batch: int specifying the batch size.
        
        Returns:
            z: latent vector with shape (n_batch, latent_dim)
        """
        return torch.randn(n_batch, self.latent_dim, device=self.device)
    
    def sample(self, n_batch=32, max_len=100, z=None, temp=1.0):
        """
        Generate n_batch samples.
        """
        self.eval()
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)   # ensure that z is on the same device as the model
            z_0 = z.unsqueeze(1)

            # get initial hidden state
            h = self.latent2hidden(z)   # (batch_size, hidden_dim)

            # repeat initial hidden state along the number of GRU layers
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)    # (num_gru_layers, batch_size, hidden_dim)

            # define first token
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)

            # initialize tensor to store generated sequences
            x_gen = torch.tensor([self.pad], device=self.device).repeat(n_batch, max_len)
            x_gen[:, 0] = self.bos

            # iteratively generate sequences
            for t in range(1, max_len):
                # embed the current token and concatenate it to the latent vector
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                # get the output, apply temperature scaling to the logits, and get probabilities
                output, h = self.decoder_rnn(x_input, h)
                logits = self.hidden2vocab(output.squeeze(1))
                probs = F.softmax(logits / temp, dim=-1)

                # sample the next token
                w = torch.multinomial(probs, num_samples=1).squeeze(1)
                x_gen[:, t] = w

        return x_gen
