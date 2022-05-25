from auxiliar import evaluate_test

wer_list = []
# evaluate_test("/workspace/models/it/wav2vec2model", "/workspace/models/it/LM/LM.bin",
#               "multilingual_librispeech", "italian", "italian", alpha=0.1, beta=0.5)
#
# evaluate_test("/workspace/models/pl/wav2vec2model", "/workspace/models/pl/LM/LM.bin",
#               "multilingual_librispeech", "polish", "polish", alpha=0.1, beta=0.5)

wer_list.append(evaluate_test("/workspace/models/de/wav2vec2model", "/workspace/models/de/LM/LM.bin",
                              "multilingual_librispeech", "german", "german", alpha=0.0, beta=0.0))
wer_list.append(evaluate_test("/workspace/models/de/wav2vec2model", "/workspace/models/de/LM/LM.bin",
                              "multilingual_librispeech", "german", "german", alpha=0.2, beta=0.5))
wer_list.append(evaluate_test("/workspace/models/de/wav2vec2model", "/workspace/models/de/LM/LM.bin",
                              "multilingual_librispeech", "german", "german", alpha=0.4, beta=1))
wer_list.append(evaluate_test("/workspace/models/de/wav2vec2model", "/workspace/models/de/LM/LM.bin",
                              "multilingual_librispeech", "german", "german", alpha=0.6, beta=1))
wer_list.append(evaluate_test("/workspace/models/de/wav2vec2model", "/workspace/models/de/LM/LM.bin",
                              "multilingual_librispeech", "german", "german", alpha=1, beta=1.5))
wer_list.append(evaluate_test("/workspace/models/de/wav2vec2model", "/workspace/models/de/LM/LM.bin",
                              "multilingual_librispeech", "german", "german", alpha=1.5, beta=1.5))

# evaluate_test("/workspace/models/pt/wav2vec2model", "/workspace/models/pt/LM/LM.bin",
#               "multilingual_librispeech", "portuguese", "portuguese", alpha=0.1, beta=0.5)
#
# evaluate_test("/workspace/models/fr/wav2vec2model", "/workspace/models/fr/LM/LM.bin",
#               "multilingual_librispeech", "french", "french", alpha=0.1, beta=0.5)
#
# evaluate_test("/workspace/models/es/wav2vec2model", "/workspace/models/es/LM/LM.bin",
#               "multilingual_librispeech", "spanish", "spanish", alpha=0.1, beta=0.5)
#
# evaluate_test("/workspace/models/en/wav2vec2model", "/workspace/models/en/LM/LM.bin",
#               "common_voice", "en", "english", alpha=0.1, beta=0.5)




print("\n\n\n\n\n\n\n")
for item in wer_list:
    print("\n************************************************\n")
    print("\nLANGUAGE: " + item[0] + ", WER: {:2f}".format(item[1]))
    print("\n************************************************\n\n")

