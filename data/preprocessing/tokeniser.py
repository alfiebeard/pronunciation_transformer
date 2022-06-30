def tokenise(ipa, remove_zero_width_space=True):
    """
    Tokenise a string of IPA characters, e.g., "ʃˈɔːt‍ʃe‍ɪnd‍ʒd" -> ['ʃ', 'ɔː', 'tʃ', 'eɪ', 'n', 'd', 'ʒ', 'd'].
    This drops IPA tokens we don't use in our model.
    """

    # List of all phonemes and map special cases over
    phonemes = ['b', 'd', 'g', 'm', 'n', 'p', 't', 'ɹ', 'k', 'tʃ', 'f', 'ʤ', 'l', 's', 'v', 'z', 'h', 'ŋ', 'ʃ', 'ʒ', 'θ', 'ð', 'w', 'j', 'æ', 'e', 'ɪ', 'ɒ', 'ʌ', 'ʊ', 'ə', 'ɑː', 'ɜː', 'ɔː', 'uː', 'iː', 'eɪ', 'aɪ', 'əʊ', 'aʊ', 'ɔɪ', 'eə', 'ɪə', 'ʊə']
    special_cases = {'ɛ': 'e', 'ɐ': 'ə', 'i': 'iː', 'ɔ': 'ɔː'}  # Mappings in special cases to ensure matching phonemes
    ignore = ["ˈ", "ˌ"]
    tokens = []

    if remove_zero_width_space:
        ipa = ipa.replace('\u200d', '')

    symbol_i = 0
    ipa_length = len(ipa)

    # Loop through ipa string
    while symbol_i < ipa_length:
        # If not last symbol
        if symbol_i < ipa_length - 1:
            # Check next two symbols for phoneme - in the case it's a two character phoneme
            phoneme = ipa[symbol_i: symbol_i + 2]
            if phoneme in phonemes:
                tokens.append(phoneme)
                symbol_i += 2
                continue
        
        # Otherwise it's only a one symbol phoneme
        if ipa[symbol_i] in phonemes:
            # If symbol exists, add to output
            tokens.append(ipa[symbol_i])
        elif ipa[symbol_i] in special_cases:
            tokens.append(special_cases[ipa[symbol_i]])
        else:
            if ipa[symbol_i] not in ignore:
                print(ipa)
                print('Skipping missing token {0}'.format(ipa[symbol_i]))
        symbol_i += 1

    return tokens


if __name__ == '__main__':
    print("Tokenise ʃˈɔːt‍ʃe‍ɪnd‍ʒd:")
    print(tokenise("ʃˈɔːt‍ʃe‍ɪnd‍ʒd"))
