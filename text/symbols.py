""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]
_japanese = ['ky','sp', 'sh', 'ch', 'ts','ty', 'ry', 'ny', 'by', 'hy', 'gy', 'kw', 'gw', 'kj', 'gj', 'my', 'py','dy']
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]
# Add phoneme of ipa format
_ipa = ['w', 'iː', 'θ', 'ɔː', 't', 'ɜː', 'l', 'ʊ', 'k', 'ɪ', 'ŋ', 'n', 'ɡ', 'd', 'ʃ', 'eɪ', 'p', 'ð', 'ɑː', 'aɪ', 'ɛ', 's', 'eə', 'ɹ', 'ə', 'j', 'uː', 'h', 'aʊ', 'a', 'ɒ', 'v', 'm', 'ɐ', 'z', 'b', 'ʌ', 'i', 'f', 'əʊ', 'ʊə', 'əl', 'sp', 'iə', 'dʒ', 'ʒ', 'tʃ', 'ɔɪ', 'aɪə', 'n̩']

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
    + _japanese
    + _ipa
)
