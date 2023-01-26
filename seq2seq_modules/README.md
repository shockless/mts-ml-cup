# Friendly reminder

1. It's ```[batch_size, seq_len, embedding_dim]``` not ```[seq_len, batch_size, embedding_dim]``` <br>

2. Right now preprocessor is pretty slow (However O(n), but converts pd to np). Want to optimize - go for it :)