


encoder = Encoder(4, 32, 32)
decoder = Decoder()
policy = Policy(state, encoder, decoder)

policy.forward()